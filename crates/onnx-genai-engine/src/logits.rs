//! Logit processor chain.
//!
//! Mandatory bias, constraints and stop checks surround a declared ordered
//! sampler pipeline. Default stages follow llama.cpp ordering; Mirostat owns
//! its own temperature/selector mode. Device shortcuts require a cached typed
//! transform proof, including actual parameters and equivalent stage order.

use anyhow::Context as _;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Mutex;

use crate::config::MirostatVersion;
use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::{ApproximateTokEnv, InferenceCapabilities, TokRxInfo, TokTrie};
use llguidance::{Constraint as LlgConstraint, ParserFactory};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Token id used by the generation engine.
pub type TokenId = u32;

/// A stop sequence expressed either as generated text or token ids.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopSequence {
    /// Stop when generated text ends with this string.
    Text(String),
    /// Stop when generated token ids end with this sequence.
    Tokens(Vec<TokenId>),
}

impl StopSequence {
    fn is_empty(&self) -> bool {
        match self {
            Self::Text(text) => text.is_empty(),
            Self::Tokens(tokens) => tokens.is_empty(),
        }
    }
}

/// Context passed to logit processors.
#[derive(Debug, Clone, Default)]
pub struct ProcessorContext {
    /// Prompt token ids for the sequence.
    pub prompt_tokens: Vec<TokenId>,
    /// Tokens generated so far in this sequence.
    pub generated_tokens: Vec<TokenId>,
    /// Generated text so far, if detokenization is available.
    pub generated_text: String,
    /// Current step index.
    pub step: usize,
}

/// Non-logit side-channel emitted by processors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessorSignal {
    /// A configured stop sequence matched the current generated output.
    StopSequence { index: usize },
}

/// A candidate token considered by a constrained decoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenCandidate {
    pub token_id: TokenId,
    pub text: String,
    pub is_eos: bool,
}

/// A decoding constraint returns the mask of tokens that keep the output valid.
///
/// The processor owns pre-decoded token text so constraints can validate at the
/// character level. Decoding every vocabulary token is intentionally simple and
/// correct, but can be slow for large vocabularies.
pub trait Constraint: Send + Sync {
    /// Return exactly one decision per candidate. A grammar rejection is false;
    /// parser/history failures are errors, never a fabricated all-false mask.
    fn allowed_next_tokens(
        &self,
        context: &ProcessorContext,
        candidates: &[TokenCandidate],
    ) -> anyhow::Result<Vec<bool>>;

    fn name(&self) -> &str;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrammarConstraintKind {
    JsonSchema,
    Regex,
    Lark,
}

/// Exact transforms the fixed device sampler can perform in its declared order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeviceSamplingTransform {
    Temperature(f32),
    TopK(usize),
    TopP(f32),
    MinP(f32),
    TopKTopP { top_k: usize, top_p: f32 },
}

fn compare_candidate_logits(a: usize, b: usize, logits: &[f32]) -> std::cmp::Ordering {
    logits[a]
        .is_nan()
        .cmp(&logits[b].is_nan())
        .then_with(|| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .then_with(|| a.cmp(&b))
}

/// Logical candidate identities, allocated only for identity-sensitive stages.
/// Numeric masks alone do not remove a candidate from this set.
pub struct SamplingCandidates {
    indices: Vec<usize>,
    hard_allowed: Vec<bool>,
}
impl SamplingCandidates {
    fn new(logits: &[f32]) -> Self {
        Self {
            indices: (0..logits.len()).collect(),
            hard_allowed: logits
                .iter()
                .map(|v| !v.is_nan() && *v != f32::NEG_INFINITY)
                .collect(),
        }
    }
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }
    pub fn len(&self) -> usize {
        self.indices.len()
    }
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
    pub fn retain(&mut self, mut keep: impl FnMut(usize) -> bool) {
        self.indices.retain(|&id| keep(id));
    }
    fn top_k(&mut self, logits: &[f32], k: usize) {
        if k == 0 || k >= self.len() {
            return;
        }
        self.indices
            .sort_unstable_by(|&a, &b| compare_candidate_logits(a, b, logits));
        self.indices.truncate(k);
    }
}

/// A logit processor modifies the logit distribution before sampling.
pub trait LogitProcessor: Send + Sync {
    fn process(&self, logits: &mut [f32], context: &ProcessorContext) -> anyhow::Result<()>;
    fn name(&self) -> &str;

    /// Whether selected tokens update request-local sampler state.
    fn needs_selection_feedback(&self) -> bool {
        false
    }

    /// Stage a selected token. A later process call may rewind this choice
    /// using its generated-token prefix before retrying a branch.
    fn on_selected(&self, _token: TokenId, _context: &ProcessorContext) -> anyhow::Result<()> {
        Ok(())
    }

    fn needs_candidate_identity(&self) -> bool {
        false
    }
    /// Preserve identities by default; a compacting stage must describe removal.
    fn process_with_candidates(
        &self,
        logits: &mut [f32],
        context: &ProcessorContext,
        candidates: &mut SamplingCandidates,
    ) -> anyhow::Result<()> {
        let count = self.process_with_candidate_count(logits, context, candidates.len())?;
        anyhow::ensure!(
            count == candidates.len(),
            "sampling stage '{}' changed candidate count from {} to {count} without an identity contract; implement process_with_candidates for this stage",
            self.name(),
            candidates.len()
        );
        Ok(())
    }

    /// Whether this processor needs logical candidate cardinality after truncation.
    fn needs_candidate_count(&self) -> bool {
        false
    }

    /// Process logits and carry the logical candidate-array size. Masking alone
    /// preserves the size; truncating samplers override this to describe removal.
    fn process_with_candidate_count(
        &self,
        logits: &mut [f32],
        context: &ProcessorContext,
        count: usize,
    ) -> anyhow::Result<usize> {
        self.process(logits, context)?;
        Ok(count)
    }

    /// Declare exact fixed-device semantics; unknown/custom transforms stay on host.
    fn device_sampling_transform(&self) -> Option<DeviceSamplingTransform> {
        None
    }

    /// Return a termination signal for the current context, if this processor owns one.
    fn signal(&self, _context: &ProcessorContext) -> Option<ProcessorSignal> {
        None
    }

    /// Whether this processor is guaranteed to leave the position of the maximum
    /// logit unchanged, for every possible input.
    ///
    /// This is what lets greedy decoding skip the chain and take the device
    /// argmax, which never materializes logits on the host at all.
    ///
    /// Rank-preserving truncations and a static zero-temperature point mass
    /// may qualify. History penalties, constraints, typical sampling and XTC
    /// can change the winning token and must execute on host before selection.
    fn preserves_argmax(&self) -> bool {
        false
    }
}

/// Ordered chain of logit processors.
pub struct ProcessorChain {
    processors: Vec<Box<dyn LogitProcessor>>,
    device_transforms: Option<Vec<DeviceSamplingTransform>>,
    track_candidates: bool,
    track_candidate_ids: bool,
    optional_start: Option<usize>,
    selection_feedback: bool,
}
impl Default for ProcessorChain {
    fn default() -> Self {
        Self::new()
    }
}

impl ProcessorChain {
    pub fn new() -> Self {
        Self {
            processors: Vec::new(),
            device_transforms: Some(Vec::new()),
            track_candidates: false,
            track_candidate_ids: false,
            optional_start: None,
            selection_feedback: false,
        }
    }

    pub fn builder() -> ProcessorChainBuilder {
        ProcessorChainBuilder::new()
    }

    pub fn add(&mut self, processor: Box<dyn LogitProcessor>) {
        self.track_candidates |= processor.needs_candidate_count();
        self.track_candidate_ids |= processor.needs_candidate_identity();
        self.selection_feedback |= processor.needs_selection_feedback();
        if self.selection_feedback {
            self.device_transforms = None;
        }
        if let Some(transforms) = &mut self.device_transforms {
            if let Some(transform) = processor.device_sampling_transform() {
                transforms.push(transform);
            } else {
                self.device_transforms = None;
            }
        }
        self.processors.push(processor);
    }

    pub(crate) fn begin_optional_stages(&mut self) {
        self.optional_start = Some(self.processors.len());
    }

    pub(crate) fn device_transforms(&self) -> Option<&[DeviceSamplingTransform]> {
        self.device_transforms.as_deref()
    }

    pub(crate) fn fuse_last_top_k_with_top_p(&mut self, top_p: f32) -> bool {
        let Some(DeviceSamplingTransform::TopK(top_k)) = self
            .processors
            .last()
            .and_then(|processor| processor.device_sampling_transform())
        else {
            return false;
        };
        self.processors.pop();
        self.processors
            .push(Box::new(TopKTopPProcessor { top_k, top_p }));
        self.device_transforms = if self.selection_feedback {
            None
        } else {
            self.processors
                .iter()
                .try_fold(Vec::new(), |mut transforms, processor| {
                    transforms.push(processor.device_sampling_transform()?);
                    Some(transforms)
                })
        };
        true
    }

    /// Whether the chain has no processors. When empty, greedy decoding reduces
    /// to a plain argmax of the raw logits, enabling the decode fast path that
    /// selects the token on the device/buffer without materializing the full
    /// vocabulary on the host.
    pub fn is_empty(&self) -> bool {
        self.processors.is_empty()
    }

    /// Whether greedy selection over the raw logits gives the same token as
    /// greedy selection over this chain's output.
    ///
    /// An empty chain trivially qualifies, and so does a greedy request that
    /// only carries stop sequences -- which is what a chat template leaves
    /// behind once the sampling warpers are dropped. Requiring literal
    /// emptiness here denied the device argmax to every model shipping a chat
    /// template, which is nearly all of them.
    pub fn preserves_argmax(&self) -> bool {
        !self.selection_feedback
            && self
                .processors
                .iter()
                .all(|processor| processor.preserves_argmax())
    }

    pub fn add_constraint(
        &mut self,
        constraint: Box<dyn Constraint>,
        token_texts: Vec<Option<String>>,
        eos_token_ids: Vec<TokenId>,
    ) {
        self.add(Box::new(ConstraintProcessor::new(
            constraint,
            token_texts,
            eos_token_ids,
        )));
    }

    /// Apply processors in insertion order.
    pub fn process(&self, logits: &mut [f32], context: &ProcessorContext) -> anyhow::Result<()> {
        if self.track_candidate_ids {
            let (mandatory, optional) = self.processors.split_at(self.optional_start.unwrap_or(0));
            for processor in mandatory {
                processor
                    .process(logits, context)
                    .with_context(|| format!("mandatory stage '{}' failed", processor.name()))?;
            }
            let mut candidates = SamplingCandidates::new(logits);
            for processor in optional {
                processor
                    .process_with_candidates(logits, context, &mut candidates)
                    .with_context(|| {
                        format!(
                            "sampling stage '{}' failed at generated token {}",
                            processor.name(),
                            context.generated_tokens.len()
                        )
                    })?;
            }
        } else if self.track_candidates {
            let mut candidates = logits.len();
            for processor in &self.processors {
                candidates = processor
                    .process_with_candidate_count(logits, context, candidates)
                    .with_context(|| {
                        format!(
                            "sampling stage '{}' failed at generated token {}",
                            processor.name(),
                            context.generated_tokens.len()
                        )
                    })?;
            }
        } else {
            for processor in &self.processors {
                processor.process(logits, context).with_context(|| {
                    format!(
                        "sampling stage '{}' failed at generated token {}",
                        processor.name(),
                        context.generated_tokens.len()
                    )
                })?;
            }
        }
        Ok(())
    }

    /// Stage feedback only after processing and selecting against this context.
    /// Keep generated history unchanged until this method succeeds.
    pub fn on_selected(&self, token: TokenId, context: &ProcessorContext) -> anyhow::Result<()> {
        if self.selection_feedback {
            for processor in &self.processors {
                processor.on_selected(token, context).with_context(|| format!(
                    "sampling stage '{}' could not stage selected token {token} at generated token {}",
                    processor.name(), context.generated_tokens.len()
                ))?;
            }
        }
        Ok(())
    }

    /// Return the first termination signal from the ordered chain.
    pub fn signal(&self, context: &ProcessorContext) -> Option<ProcessorSignal> {
        self.processors.iter().find_map(|proc| proc.signal(context))
    }

    /// Processor names in configured order, useful for diagnostics and tests.
    pub fn names(&self) -> Vec<&str> {
        self.processors.iter().map(|proc| proc.name()).collect()
    }
}

/// Builder/registry for assembling custom processor chains.
#[derive(Default)]
pub struct ProcessorChainBuilder {
    chain: ProcessorChain,
}

impl ProcessorChainBuilder {
    pub fn new() -> Self {
        Self {
            chain: ProcessorChain::new(),
        }
    }

    pub fn register_processor(mut self, processor: Box<dyn LogitProcessor>) -> Self {
        self.chain.add(processor);
        self
    }

    pub fn register_constraint(
        mut self,
        constraint: Box<dyn Constraint>,
        token_texts: Vec<Option<String>>,
        eos_token_ids: Vec<TokenId>,
    ) -> Self {
        self.chain
            .add_constraint(constraint, token_texts, eos_token_ids);
        self
    }

    pub fn build(self) -> ProcessorChain {
        self.chain
    }
}

// --- Built-in processors ---

pub struct ConstraintProcessor {
    constraint: Box<dyn Constraint>,
    token_texts: Vec<Option<String>>,
    /// Every id the model ends on, so a constrained decode treats each of them
    /// as terminal rather than only the one a caller happened to name.
    eos_token_ids: Vec<TokenId>,
}

impl ConstraintProcessor {
    pub fn new(
        constraint: Box<dyn Constraint>,
        token_texts: Vec<Option<String>>,
        eos_token_ids: Vec<TokenId>,
    ) -> Self {
        Self {
            constraint,
            token_texts,
            eos_token_ids,
        }
    }
}

impl LogitProcessor for ConstraintProcessor {
    fn process(&self, logits: &mut [f32], context: &ProcessorContext) -> anyhow::Result<()> {
        let candidates: Vec<_> = (0..logits.len())
            .map(|idx| {
                let token_id = idx as TokenId;
                TokenCandidate {
                    token_id,
                    text: self
                        .token_texts
                        .get(idx)
                        .and_then(|text| text.clone())
                        .unwrap_or_default(),
                    is_eos: self.eos_token_ids.contains(&token_id),
                }
            })
            .collect();
        let mask = self
            .constraint
            .allowed_next_tokens(context, &candidates)
            .with_context(|| {
                format!(
                    "{} could not compute a token mask at generated offset {}",
                    self.constraint.name(),
                    context.generated_tokens.len()
                )
            })?;
        anyhow::ensure!(
            mask.len() == logits.len(),
            "{} returned {} mask entries for {} candidate logits; repair the constraint implementation so every candidate has one decision",
            self.constraint.name(),
            mask.len(),
            logits.len()
        );
        for (idx, logit) in logits.iter_mut().enumerate() {
            if !mask.get(idx).copied().unwrap_or(false) {
                *logit = f32::NEG_INFINITY;
            }
        }
        anyhow::ensure!(
            logits.iter().any(|logit| logit.is_finite()),
            "{} leaves no finite selectable token at generated offset {}; verify the tokenizer vocabulary can continue the grammar and relax incompatible token bans or constraints",
            self.constraint.name(),
            context.generated_tokens.len()
        );
        Ok(())
    }

    fn name(&self) -> &str {
        self.constraint.name()
    }
}

/// Character-level JSON grammar constraint.
///
/// This validates the generated prefix plus each candidate token and only allows
/// tokens that keep the prefix valid and completable. EOS is allowed only after
/// a complete, balanced JSON value. JSON Schema uses the llguidance recognizer.
#[derive(Debug, Clone, Default)]
pub struct JsonConstraint {
    prefix: String,
}

impl JsonConstraint {
    /// Seed the recognizer with existing output, leaving new-token history separate.
    pub fn new(prefix: String) -> anyhow::Result<Self> {
        anyhow::ensure!(
            Self::prefix_is_valid(&prefix),
            "JSON constraint prefix is not a valid partial JSON value; continue a syntactically valid JSON prefix or omit constrained JSON output"
        );
        Ok(Self { prefix })
    }

    pub fn prefix_is_valid(text: &str) -> bool {
        JsonPrefixParser::parse(text).is_ok()
    }

    pub fn is_complete(text: &str) -> bool {
        JsonPrefixParser::parse(text).is_ok_and(|parser| parser.is_complete())
    }
}

impl Constraint for JsonConstraint {
    fn allowed_next_tokens(
        &self,
        context: &ProcessorContext,
        candidates: &[TokenCandidate],
    ) -> anyhow::Result<Vec<bool>> {
        let generated = if self.prefix.is_empty() {
            std::borrow::Cow::Borrowed(context.generated_text.as_str())
        } else {
            std::borrow::Cow::Owned(format!("{}{}", self.prefix, context.generated_text))
        };
        let complete = Self::is_complete(&generated);
        Ok(candidates
            .iter()
            .map(|candidate| {
                if candidate.is_eos {
                    return complete;
                }

                if candidate.text.is_empty() {
                    return false;
                }
                let mut next = String::with_capacity(generated.len() + candidate.text.len());
                next.push_str(&generated);
                next.push_str(&candidate.text);
                Self::prefix_is_valid(&next)
            })
            .collect())
    }

    fn name(&self) -> &str {
        "json_constraint"
    }
}

/// Existing output could not be admitted into a requested grammar.
#[derive(Debug, thiserror::Error)]
#[error(
    "existing assistant prefix is rejected by the declared grammar: {source:#}; continue a prefix accepted by this grammar or change the requested grammar"
)]
pub struct ConstraintPrefixError {
    #[source]
    source: anyhow::Error,
}

/// llguidance-backed grammar constraint for JSON Schema, regex, and Lark grammars.
///
/// The state machine is advanced from `ProcessorContext::generated_tokens` before
/// computing the next-token mask. This keeps the existing logit-processor API
/// unchanged while still using llguidance's compute-mask/commit-token loop.
pub struct LlguidanceConstraint {
    kind: GrammarConstraintKind,
    sampled_vocab_size: usize,
    inner: Mutex<LlguidanceState>,
}

struct LlguidanceState {
    constraint: LlgConstraint,
    initial: LlgConstraint,
    committed_tokens: Vec<TokenId>,
}

impl LlguidanceConstraint {
    pub fn from_hf_tokenizer(
        kind: GrammarConstraintKind,
        grammar: &str,
        tokenizer: &tokenizers::Tokenizer,
        vocab_size: usize,
        eos_token_id: Option<TokenId>,
    ) -> anyhow::Result<Self> {
        let mut byte_tokenizer =
            toktrie_hf_tokenizers::ByteTokenizer::from_tokenizer(tokenizer.clone())?;
        if let Some(eos_token_id) = eos_token_id {
            byte_tokenizer.set_eos_token(eos_token_id);
        }
        let tok_env = byte_tokenizer.into_tok_env(Some(vocab_size))?;
        Self::from_tok_env(kind, grammar, &tok_env, &[], vocab_size)
    }

    pub fn from_token_texts(
        kind: GrammarConstraintKind,
        grammar: &str,
        token_texts: &[Option<String>],
        eos_token_id: Option<TokenId>,
    ) -> anyhow::Result<Self> {
        let token_bytes = token_texts
            .iter()
            .map(|text| text.as_deref().unwrap_or_default().as_bytes().to_vec())
            .collect();
        Self::from_token_bytes(
            kind,
            grammar,
            token_bytes,
            &eos_token_id.into_iter().collect::<Vec<_>>(),
            "",
        )
    }

    /// Consume existing text exactly, independently of first-token normalization
    /// or vocabulary boundaries. The extra grammar-only token is never exposed
    /// to model sampling or counted as newly generated output.
    pub(crate) fn from_token_bytes_with_prefix(
        kind: GrammarConstraintKind,
        grammar: &str,
        token_bytes: Vec<Vec<u8>>,
        endings: &[TokenId],
        prefix: &str,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !prefix.is_empty(),
            "prefilled grammar construction requires nonempty existing text"
        );
        // The schema compiler starts at the JSON value rather than its
        // optional leading RFC 8259 whitespace. Preserve that whitespace in
        // the actual prompt, but enter the value recognizer after it.
        let value_prefix = match kind {
            GrammarConstraintKind::JsonSchema => prefix.trim_start_matches([' ', '\t', '\r', '\n']),
            _ => prefix,
        };
        Self::from_token_bytes(kind, grammar, token_bytes, endings, value_prefix)
    }

    fn from_token_bytes(
        kind: GrammarConstraintKind,
        grammar: &str,
        mut token_bytes: Vec<Vec<u8>>,
        endings: &[TokenId],
        prefix: &str,
    ) -> anyhow::Result<Self> {
        let sampled_vocab_size = token_bytes.len();
        anyhow::ensure!(
            !token_bytes.is_empty(),
            "grammar vocabulary is empty; load a tokenizer with declared token byte pieces"
        );
        for &ending in endings {
            let bytes=token_bytes.get_mut(usize::try_from(ending)?)
                .with_context(||format!("grammar ending token {ending} is outside its tokenizer vocabulary; repair the declared EOS set"))?;
            *bytes = vec![TokTrie::SPECIAL_TOKEN_MARKER];
            bytes.extend_from_slice(b"<eos>");
        }
        // Bound trie depth independently of request length. Bytes can split a
        // UTF-8 scalar here, just as real byte-fallback vocabulary pieces do.
        const PREFIX_CHUNK_BYTES: usize = 256;
        let mut prefix_ids = Vec::new();
        for chunk in prefix.as_bytes().chunks(PREFIX_CHUNK_BYTES) {
            let id = TokenId::try_from(token_bytes.len())
                .context("grammar vocabulary has no room for an existing-prefix token")?;
            token_bytes.push(chunk.to_vec());
            prefix_ids.push(id);
        }
        // A tokenizer without a declared ending must not silently reserve its
        // ordinary token 0 as EOS. This recognizer-only ending is never sampled.
        let eos = match endings.first().copied() {
            Some(id) => id,
            None => {
                let id = TokenId::try_from(token_bytes.len())
                    .context("grammar vocabulary has no room for an internal ending")?;
                let mut bytes = vec![TokTrie::SPECIAL_TOKEN_MARKER];
                bytes.extend_from_slice(b"<eos>");
                token_bytes.push(bytes);
                id
            }
        };
        let total_bytes = token_bytes.iter().try_fold(0usize, |sum, bytes| {
            sum.checked_add(bytes.len())
                .context("grammar byte vocabulary size overflow")
        })?;
        anyhow::ensure!(
            u32::try_from(total_bytes).is_ok(),
            "grammar byte vocabulary exceeds u32 offsets; shorten the existing prefix or use a smaller tokenizer vocabulary"
        );
        // toktrie 1.7 stores ids in 24 bits and reserves the all-ones id.
        // Check this wire-format capacity before an id can be truncated.
        const TOKTRIE_VOCAB_CAPACITY: usize = 0x00ff_ffff;
        anyhow::ensure!(
            token_bytes.len() <= TOKTRIE_VOCAB_CAPACITY,
            "grammar vocabulary including prefix cells exceeds toktrie's 24-bit token capacity; shorten the prefix or use a tokenizer within the grammar backend's supported vocabulary range"
        );
        let info = TokRxInfo {
            vocab_size: u32::try_from(token_bytes.len())
                .context("grammar vocabulary exceeds u32 token ids")?,
            tok_eos: eos,
            tok_bos: None,
            tok_pad: None,
            tok_unk: None,
            tok_end_of_turn: None,
        };
        let tok_trie = std::panic::catch_unwind(std::panic::AssertUnwindSafe(||TokTrie::from(&info, &token_bytes)))
            .map_err(|_|anyhow::anyhow!("grammar byte vocabulary exceeds the tokenizer trie's supported limits; shorten the existing prefix or repair oversized tokenizer pieces"))?;
        let tok_trie = if endings.len() > 1 {
            tok_trie.with_eos_tokens(endings)
        } else {
            tok_trie
        };
        let tok_env: llguidance::toktrie::TokEnv =
            std::sync::Arc::new(ApproximateTokEnv::new(tok_trie));
        Self::from_tok_env(kind, grammar, &tok_env, &prefix_ids, sampled_vocab_size)
    }

    fn from_tok_env(
        kind: GrammarConstraintKind,
        grammar: &str,
        tok_env: &llguidance::toktrie::TokEnv,
        prefix_ids: &[TokenId],
        sampled_vocab_size: usize,
    ) -> anyhow::Result<Self> {
        let grammar = top_level_grammar(kind, grammar)?;
        let factory = ParserFactory::new(tok_env, InferenceCapabilities::default(), &[])?;
        let parser = factory.create_parser(grammar)?;
        let mut constraint = LlgConstraint::new(parser);
        if !prefix_ids.is_empty() {
            constraint.start_without_prompt();
            constraint
                .force_tokens(prefix_ids)
                .map_err(|source| ConstraintPrefixError { source })?;
        }
        Ok(Self {
            kind,
            sampled_vocab_size,
            inner: Mutex::new(LlguidanceState {
                initial: constraint.deep_clone(),
                constraint,
                committed_tokens: Vec::new(),
            }),
        })
    }
}

fn top_level_grammar(
    kind: GrammarConstraintKind,
    grammar: &str,
) -> anyhow::Result<TopLevelGrammar> {
    match kind {
        GrammarConstraintKind::JsonSchema => {
            let schema = serde_json::from_str(grammar)?;
            Ok(TopLevelGrammar::from_json_schema(schema))
        }
        GrammarConstraintKind::Regex => Ok(TopLevelGrammar::from_regex(grammar)),
        GrammarConstraintKind::Lark => Ok(TopLevelGrammar::from_lark(grammar.to_string())),
    }
}

impl Constraint for LlguidanceConstraint {
    fn allowed_next_tokens(
        &self,
        context: &ProcessorContext,
        candidates: &[TokenCandidate],
    ) -> anyhow::Result<Vec<bool>> {
        let mut inner = self.inner.lock().map_err(|_| {
            anyhow::anyhow!(
                "{} parser lock is poisoned; restart this generation with a fresh grammar state",
                self.name()
            )
        })?;
        if !context
            .generated_tokens
            .starts_with(&inner.committed_tokens)
        {
            inner.reset();
        }
        let result = (|| -> anyhow::Result<Vec<bool>> {
            for &token in &context.generated_tokens[inner.committed_tokens.len()..] {
                anyhow::ensure!(
                    (token as usize) < self.sampled_vocab_size,
                    "grammar history token {token} is outside the sampled vocabulary of {} entries; repair generated history or its tokenizer",
                    self.sampled_vocab_size
                );
                let offset = inner.committed_tokens.len();
                let step = inner.current_step().with_context(|| {
                    format!(
                        "prepare {} before committing history token {token} at offset {offset}",
                        self.name()
                    )
                })?;
                anyhow::ensure!(
                    !step.is_stop() || inner.constraint.tok_trie().eos_tokens().contains(&token),
                    "{} history token {token} at offset {offset} continues after the grammar completed; truncate the stale suffix or start a new generation",
                    self.name()
                );
                let commit = inner
                    .constraint
                    .commit_token(Some(token))
                    .with_context(|| {
                        format!(
                            "commit history token {token} at offset {} into {}",
                            inner.committed_tokens.len(),
                            self.name()
                        )
                    })?;
                anyhow::ensure!(
                    commit.backtrack == 0
                        && (commit.ff_tokens.is_empty() || commit.ff_tokens == [token]),
                    "{} requested unsupported history splicing at token {token}: backtrack={}, forced_tokens={:?}; use a grammar without hidden stop captures or token splices",
                    self.name(),
                    commit.backtrack,
                    commit.ff_tokens
                );
                inner.committed_tokens.push(token);
            }
            let step = inner.current_step().with_context(|| {
                format!(
                    "compute {} after {} generated tokens",
                    self.name(),
                    context.generated_tokens.len()
                )
            })?;
            if step.is_stop() {
                return Ok(candidates
                    .iter()
                    .map(|candidate| {
                        (candidate.token_id as usize) < self.sampled_vocab_size && candidate.is_eos
                    })
                    .collect());
            }
            let mask=step.sample_mask.as_ref().with_context(||format!("{} returned neither a stop nor a sampling mask; use a grammar compatible with ordinary token sampling",self.name()))?;
            Ok(candidates
                .iter()
                .map(|candidate| {
                    let idx = candidate.token_id as usize;
                    idx < self.sampled_vocab_size && idx < mask.len() && mask.get(idx)
                })
                .collect())
        })();
        if result.is_err() {
            inner.reset();
        }
        result
    }

    fn name(&self) -> &str {
        match self.kind {
            GrammarConstraintKind::JsonSchema => "json_schema_constraint",
            GrammarConstraintKind::Regex => "regex_constraint",
            GrammarConstraintKind::Lark => "lark_constraint",
        }
    }
}

impl LlguidanceState {
    fn current_step(&mut self) -> anyhow::Result<llguidance::toktrie::StepResult> {
        let previous = self.constraint.step_result();
        if previous.is_stop() || previous.sample_mask.is_some() {
            return Ok(previous.clone());
        }
        Ok(self.constraint.compute_mask()?.clone())
    }

    fn reset(&mut self) {
        self.constraint = self.initial.deep_clone();
        self.committed_tokens.clear();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContainerKind {
    Object,
    Array,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Expect {
    KeyOrEnd,
    Key,
    Colon,
    ValueOrEnd,
    Value,
    CommaOrEnd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Container {
    kind: ContainerKind,
    expect: Expect,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StringRole {
    Key,
    Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Mode {
    Normal,
    String {
        role: StringRole,
        escape: bool,
        unicode_remaining: u8,
    },
    Number(NumberState),
    Literal {
        target: &'static str,
        matched: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NumberPhase {
    Minus,
    Zero,
    Int,
    Dot,
    Fraction,
    ExpStart,
    ExpSign,
    ExpDigits,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NumberState {
    phase: NumberPhase,
}

impl NumberState {
    fn start(ch: char) -> Option<Self> {
        let phase = match ch {
            '-' => NumberPhase::Minus,
            '0' => NumberPhase::Zero,
            '1'..='9' => NumberPhase::Int,
            _ => return None,
        };
        Some(Self { phase })
    }

    fn is_complete(self) -> bool {
        matches!(
            self.phase,
            NumberPhase::Zero | NumberPhase::Int | NumberPhase::Fraction | NumberPhase::ExpDigits
        )
    }

    fn consume(&mut self, ch: char) -> bool {
        self.phase = match (self.phase, ch) {
            (NumberPhase::Minus, '0') => NumberPhase::Zero,
            (NumberPhase::Minus, '1'..='9') => NumberPhase::Int,
            (NumberPhase::Zero, '.') | (NumberPhase::Int, '.') => NumberPhase::Dot,
            (NumberPhase::Zero, 'e' | 'E')
            | (NumberPhase::Int, 'e' | 'E')
            | (NumberPhase::Fraction, 'e' | 'E') => NumberPhase::ExpStart,
            (NumberPhase::Int, '0'..='9') => NumberPhase::Int,
            (NumberPhase::Dot, '0'..='9') | (NumberPhase::Fraction, '0'..='9') => {
                NumberPhase::Fraction
            }
            (NumberPhase::ExpStart, '+' | '-') => NumberPhase::ExpSign,
            (NumberPhase::ExpStart, '0'..='9')
            | (NumberPhase::ExpSign, '0'..='9')
            | (NumberPhase::ExpDigits, '0'..='9') => NumberPhase::ExpDigits,
            _ => return false,
        };
        true
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct JsonPrefixParser {
    stack: Vec<Container>,
    root_complete: bool,
    mode: Mode,
}

impl JsonPrefixParser {
    fn parse(text: &str) -> Result<Self, ()> {
        let mut parser = Self {
            stack: Vec::new(),
            root_complete: false,
            mode: Mode::Normal,
        };
        for ch in text.chars() {
            parser.consume(ch)?;
        }
        Ok(parser)
    }

    fn is_complete(&self) -> bool {
        let mut parser = self.clone();
        match parser.mode {
            Mode::Normal => {}
            Mode::Number(state) if state.is_complete() => {
                parser.mode = Mode::Normal;
                if parser.finish_value().is_err() {
                    return false;
                }
            }
            _ => return false,
        }
        parser.stack.is_empty() && parser.root_complete
    }

    fn consume(&mut self, ch: char) -> Result<(), ()> {
        loop {
            match &mut self.mode {
                Mode::Normal => return self.consume_normal(ch),
                Mode::String {
                    role,
                    escape,
                    unicode_remaining,
                } => {
                    if *unicode_remaining > 0 {
                        if ch.is_ascii_hexdigit() {
                            *unicode_remaining -= 1;
                            return Ok(());
                        }
                        return Err(());
                    }
                    if *escape {
                        if matches!(ch, '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't') {
                            *escape = false;
                            return Ok(());
                        }
                        if ch == 'u' {
                            *escape = false;
                            *unicode_remaining = 4;
                            return Ok(());
                        }
                        return Err(());
                    }
                    match ch {
                        '"' => {
                            let role = *role;
                            self.mode = Mode::Normal;
                            match role {
                                StringRole::Key => {
                                    let top = self.stack.last_mut().ok_or(())?;
                                    if top.kind != ContainerKind::Object
                                        || !matches!(top.expect, Expect::Key | Expect::KeyOrEnd)
                                    {
                                        return Err(());
                                    }
                                    top.expect = Expect::Colon;
                                }
                                StringRole::Value => self.finish_value()?,
                            }
                            return Ok(());
                        }
                        '\\' => {
                            *escape = true;
                            return Ok(());
                        }
                        c if c <= '\u{1f}' => return Err(()),
                        _ => return Ok(()),
                    }
                }
                Mode::Number(state) => {
                    if state.consume(ch) {
                        return Ok(());
                    }
                    if state.is_complete() {
                        self.mode = Mode::Normal;
                        self.finish_value()?;
                        continue;
                    }
                    return Err(());
                }
                Mode::Literal { target, matched } => {
                    let expected = target.as_bytes().get(*matched).copied();
                    if expected == Some(ch as u8) {
                        *matched += 1;
                        if *matched == target.len() {
                            self.mode = Mode::Normal;
                            self.finish_value()?;
                        }
                        return Ok(());
                    }
                    if *matched == target.len() {
                        self.mode = Mode::Normal;
                        self.finish_value()?;
                        continue;
                    }
                    return Err(());
                }
            }
        }
    }

    fn consume_normal(&mut self, ch: char) -> Result<(), ()> {
        if ch.is_whitespace() {
            return Ok(());
        }
        if self.root_complete && self.stack.is_empty() {
            return Err(());
        }

        match self.current_expect() {
            Expect::Value | Expect::ValueOrEnd => {
                if matches!(self.current_expect(), Expect::ValueOrEnd) && ch == ']' {
                    return self.close_container(ContainerKind::Array);
                }
                self.start_value(ch)
            }
            Expect::KeyOrEnd => match ch {
                '}' => self.close_container(ContainerKind::Object),
                '"' => {
                    if let Some(top) = self.stack.last_mut() {
                        top.expect = Expect::Key;
                    }
                    self.mode = Mode::String {
                        role: StringRole::Key,
                        escape: false,
                        unicode_remaining: 0,
                    };
                    Ok(())
                }
                _ => Err(()),
            },
            Expect::Key => {
                if ch == '"' {
                    self.mode = Mode::String {
                        role: StringRole::Key,
                        escape: false,
                        unicode_remaining: 0,
                    };
                    Ok(())
                } else {
                    Err(())
                }
            }
            Expect::Colon => {
                if ch == ':' {
                    let top = self.stack.last_mut().ok_or(())?;
                    top.expect = Expect::Value;
                    Ok(())
                } else {
                    Err(())
                }
            }
            Expect::CommaOrEnd => match (self.stack.last().map(|c| c.kind), ch) {
                (Some(ContainerKind::Object), ',') => {
                    self.stack.last_mut().ok_or(())?.expect = Expect::Key;
                    Ok(())
                }
                (Some(ContainerKind::Object), '}') => self.close_container(ContainerKind::Object),
                (Some(ContainerKind::Array), ',') => {
                    self.stack.last_mut().ok_or(())?.expect = Expect::Value;
                    Ok(())
                }
                (Some(ContainerKind::Array), ']') => self.close_container(ContainerKind::Array),
                _ => Err(()),
            },
        }
    }

    fn current_expect(&self) -> Expect {
        self.stack
            .last()
            .map(|container| container.expect)
            .unwrap_or(Expect::Value)
    }

    fn start_value(&mut self, ch: char) -> Result<(), ()> {
        match ch {
            '{' => {
                self.stack.push(Container {
                    kind: ContainerKind::Object,
                    expect: Expect::KeyOrEnd,
                });
                Ok(())
            }
            '[' => {
                self.stack.push(Container {
                    kind: ContainerKind::Array,
                    expect: Expect::ValueOrEnd,
                });
                Ok(())
            }
            '"' => {
                self.mode = Mode::String {
                    role: StringRole::Value,
                    escape: false,
                    unicode_remaining: 0,
                };
                Ok(())
            }
            't' => {
                self.mode = Mode::Literal {
                    target: "true",
                    matched: 1,
                };
                Ok(())
            }
            'f' => {
                self.mode = Mode::Literal {
                    target: "false",
                    matched: 1,
                };
                Ok(())
            }
            'n' => {
                self.mode = Mode::Literal {
                    target: "null",
                    matched: 1,
                };
                Ok(())
            }
            '-' | '0'..='9' => {
                self.mode = Mode::Number(NumberState::start(ch).ok_or(())?);
                Ok(())
            }
            _ => Err(()),
        }
    }

    fn close_container(&mut self, kind: ContainerKind) -> Result<(), ()> {
        let container = self.stack.pop().ok_or(())?;
        if container.kind != kind {
            return Err(());
        }
        self.finish_value()
    }

    fn finish_value(&mut self) -> Result<(), ()> {
        if let Some(parent) = self.stack.last_mut() {
            match (parent.kind, parent.expect) {
                (ContainerKind::Object, Expect::Value)
                | (ContainerKind::Array, Expect::Value)
                | (ContainerKind::Array, Expect::ValueOrEnd) => {
                    parent.expect = Expect::CommaOrEnd;
                    Ok(())
                }
                _ => Err(()),
            }
        } else if !self.root_complete {
            self.root_complete = true;
            Ok(())
        } else {
            Err(())
        }
    }
}

pub struct TemperatureProcessor {
    pub temperature: f32,
    pub dynamic: Option<crate::config::DynamicTemperatureConfig>,
}

impl TemperatureProcessor {
    fn resolved_temperature(&self, logits: &[f32], candidates: usize) -> f32 {
        let Some(dynamic) = self.dynamic.filter(|config| config.range > 0.0) else {
            return self.temperature;
        };
        if candidates <= 1 {
            return 1.0;
        }
        let Some(probabilities) = softmax_probabilities(logits) else {
            return self.temperature;
        };
        let entropy: f32 = probabilities
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();
        let normalized = (entropy / (candidates as f32).ln()).clamp(0.0, 1.0);
        let minimum = (self.temperature - dynamic.range).max(0.0);
        let maximum = self.temperature + dynamic.range;
        minimum + (maximum - minimum) * normalized.powf(dynamic.exponent)
    }
}

impl LogitProcessor for TemperatureProcessor {
    fn preserves_argmax(&self) -> bool {
        self.temperature == 0.0 && self.dynamic.is_none_or(|config| config.range == 0.0)
    }

    fn device_sampling_transform(&self) -> Option<DeviceSamplingTransform> {
        if self.temperature.is_finite()
            && self.temperature > 0.0
            && self.dynamic.is_none_or(|config| config.range == 0.0)
        {
            Some(DeviceSamplingTransform::Temperature(self.temperature))
        } else {
            None
        }
    }

    fn needs_candidate_count(&self) -> bool {
        self.dynamic.is_some_and(|config| config.range > 0.0)
    }
    fn process(&self, logits: &mut [f32], context: &ProcessorContext) -> anyhow::Result<()> {
        self.process_with_candidate_count(logits, context, logits.len())?;
        Ok(())
    }
    fn process_with_candidate_count(
        &self,
        logits: &mut [f32],
        _context: &ProcessorContext,
        count: usize,
    ) -> anyhow::Result<usize> {
        let temperature = self.resolved_temperature(logits, count);
        if temperature == 0.0 {
            // A collapsed interval is a point mass on the first maximum, as
            // in llama.cpp's temperature sampler. Preserve exclusions and never
            // divide by zero when entropy underflows for a peaked distribution.
            if let Some((best, _)) = logits
                .iter()
                .enumerate()
                .filter(|(_, value)| !value.is_nan() && **value != f32::NEG_INFINITY)
                .max_by(|(a, x), (b, y)| {
                    x.partial_cmp(y)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| b.cmp(a))
                })
            {
                for (index, logit) in logits.iter_mut().enumerate() {
                    if index != best {
                        *logit = f32::NEG_INFINITY;
                    }
                }
            }
        } else if temperature.is_finite() && temperature > 0.0 && temperature != 1.0 {
            for logit in logits.iter_mut() {
                *logit /= temperature;
            }
        }
        Ok(count)
    }

    fn name(&self) -> &str {
        if self.dynamic.is_some_and(|config| config.range > 0.0) {
            "dynamic_temperature"
        } else {
            "temperature"
        }
    }
}

/// A finite additive bias or an explicit token exclusion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LogitBias {
    Add(f32),
    Ban,
}

pub struct LogitBiasProcessor {
    pub biases: std::collections::BTreeMap<TokenId, LogitBias>,
}
impl LogitProcessor for LogitBiasProcessor {
    fn process(&self, logits: &mut [f32], _context: &ProcessorContext) -> anyhow::Result<()> {
        for (id, bias) in &self.biases {
            if let Some(logit) = usize::try_from(*id).ok().and_then(|id| logits.get_mut(id)) {
                match bias {
                    LogitBias::Add(value) => *logit += value,
                    LogitBias::Ban => *logit = f32::NEG_INFINITY,
                }
            }
        }
        Ok(())
    }
    fn name(&self) -> &str {
        "logit_bias"
    }
}

/// Repetition, frequency and presence penalties share one history window.
/// The count covers prompt tokens followed by committed generated tokens.
pub struct PenaltiesProcessor {
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    /// None uses the whole history; Some(0) disables all penalties.
    pub window: Option<usize>,
}

impl Default for PenaltiesProcessor {
    fn default() -> Self {
        Self {
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            window: None,
        }
    }
}

impl LogitProcessor for PenaltiesProcessor {
    fn process(&self, logits: &mut [f32], context: &ProcessorContext) -> anyhow::Result<()> {
        if self.window == Some(0)
            || (self.repetition_penalty == 1.0
                && self.frequency_penalty == 0.0
                && self.presence_penalty == 0.0)
        {
            return Ok(());
        }
        let total = context
            .prompt_tokens
            .len()
            .saturating_add(context.generated_tokens.len());
        let skip = self.window.map_or(0, |window| total.saturating_sub(window));
        let mut counts: HashMap<TokenId, usize> = HashMap::new();
        for &token in context
            .prompt_tokens
            .iter()
            .chain(context.generated_tokens.iter())
            .skip(skip)
        {
            *counts.entry(token).or_default() += 1;
        }
        for (token, count) in counts {
            if let Some(logit) = logits.get_mut(token as usize) {
                if *logit <= 0.0 {
                    *logit *= self.repetition_penalty;
                } else {
                    *logit /= self.repetition_penalty;
                }
                *logit -= count as f32 * self.frequency_penalty + self.presence_penalty;
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "penalties"
    }
}

/// Penalizes tokens that would extend a repeated suffix from the context.
///
/// For every earlier occurrence of the current suffix, DRY considers the token
/// that followed that occurrence. Once the repeated suffix reaches
/// `allowed_length`, that continuation receives
/// `multiplier * base^(matched_length - allowed_length)`.
pub struct DryProcessor {
    pub multiplier: f32,
    pub base: f32,
    pub allowed_length: usize,
    pub sequence_breakers: Vec<Vec<TokenId>>,
    pub window: Option<usize>,
}

impl LogitProcessor for DryProcessor {
    fn process(&self, logits: &mut [f32], context: &ProcessorContext) -> anyhow::Result<()> {
        if !self.multiplier.is_finite()
            || self.multiplier <= 0.0
            || !self.base.is_finite()
            || self.base < 1.0
            || self.allowed_length == 0
        {
            return Ok(());
        }

        let tokens: Vec<_> = context
            .prompt_tokens
            .iter()
            .chain(&context.generated_tokens)
            .copied()
            .collect();
        if tokens.len() <= self.allowed_length {
            return Ok(());
        }

        let tokens = &tokens[tokens
            .len()
            .saturating_sub(self.window.unwrap_or(tokens.len()))..];
        let mut boundary = vec![false; tokens.len()];
        for breaker in self
            .sequence_breakers
            .iter()
            .filter(|breaker| !breaker.is_empty())
        {
            for (start, window) in tokens.windows(breaker.len()).enumerate() {
                if window == breaker {
                    boundary[start..start + breaker.len()].fill(true);
                }
            }
        }
        let mut penalties: HashMap<TokenId, f32> = HashMap::new();
        for continuation_index in 1..tokens.len() {
            let continuation = tokens[continuation_index];
            if boundary[continuation_index] {
                continue;
            }

            let mut matched_length = 0;
            let mut historical_index = continuation_index;
            let mut recent_index = tokens.len();
            while historical_index > 0 && recent_index > 0 {
                let historical = tokens[historical_index - 1];
                let recent = tokens[recent_index - 1];
                if historical != recent
                    || boundary[historical_index - 1]
                    || boundary[recent_index - 1]
                {
                    break;
                }
                matched_length += 1;
                historical_index -= 1;
                recent_index -= 1;
            }

            if matched_length >= self.allowed_length {
                let penalty = self.multiplier
                    * self
                        .base
                        .powi((matched_length - self.allowed_length) as i32);
                penalties
                    .entry(continuation)
                    .and_modify(|current| *current = current.max(penalty))
                    .or_insert(penalty);
            }
        }

        for (token_id, penalty) in penalties {
            if let Some(logit) = logits.get_mut(token_id as usize) {
                *logit -= penalty;
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "dry"
    }
}

pub struct StopSequenceProcessor {
    pub sequences: Vec<StopSequence>,
}

impl StopSequenceProcessor {
    pub fn new(sequences: Vec<StopSequence>) -> Self {
        Self { sequences }
    }
}

impl LogitProcessor for StopSequenceProcessor {
    fn process(&self, _logits: &mut [f32], _context: &ProcessorContext) -> anyhow::Result<()> {
        Ok(())
    }

    /// This processor never touches the logits -- it exists only for its
    /// [`signal`](Self::signal), which the loop reads through
    /// `finish_reason_after_token` on every step whether or not the chain was
    /// applied. So skipping the chain here loses nothing.
    fn preserves_argmax(&self) -> bool {
        true
    }

    fn signal(&self, context: &ProcessorContext) -> Option<ProcessorSignal> {
        self.sequences
            .iter()
            .enumerate()
            .find_map(|(index, sequence)| {
                if sequence.is_empty() {
                    return None;
                }

                let matched = match sequence {
                    StopSequence::Text(text) => context.generated_text.ends_with(text),
                    StopSequence::Tokens(tokens) => context.generated_tokens.ends_with(tokens),
                };

                matched.then_some(ProcessorSignal::StopSequence { index })
            })
    }

    fn name(&self) -> &str {
        "stop_sequence"
    }
}

/// Top-n-sigma masks logits below maximum minus n population standard deviations.
pub struct TopNSigmaProcessor {
    pub n: f32,
}
impl LogitProcessor for TopNSigmaProcessor {
    fn preserves_argmax(&self) -> bool {
        true
    }

    fn process(&self, logits: &mut [f32], _context: &ProcessorContext) -> anyhow::Result<()> {
        if self.n <= 0.0 {
            return Ok(());
        }
        let mut count = 0usize;
        let mut mean = 0.0f64;
        let mut m2 = 0.0f64;
        let mut maximum = f32::NEG_INFINITY;
        for &logit in logits.iter().filter(|logit| logit.is_finite()) {
            count += 1;
            let delta = f64::from(logit) - mean;
            mean += delta / count as f64;
            m2 += delta * (f64::from(logit) - mean);
            maximum = maximum.max(logit);
        }
        if count <= 1 {
            return Ok(());
        }
        let cutoff = f64::from(maximum) - f64::from(self.n) * (m2 / count as f64).sqrt();
        for logit in logits {
            if f64::from(*logit) < cutoff {
                *logit = f32::NEG_INFINITY;
            }
        }
        Ok(())
    }
    fn name(&self) -> &str {
        "top_n_sigma"
    }
}

pub struct TopKProcessor {
    pub top_k: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct TopKLogit(f32);

impl Eq for TopKLogit {}

impl Ord for TopKLogit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl PartialOrd for TopKLogit {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl LogitProcessor for TopKProcessor {
    fn process_with_candidates(
        &self,
        logits: &mut [f32],
        context: &ProcessorContext,
        candidates: &mut SamplingCandidates,
    ) -> anyhow::Result<()> {
        candidates.top_k(logits, self.top_k);
        self.process(logits, context)
    }

    fn preserves_argmax(&self) -> bool {
        true
    }

    fn process_with_candidate_count(
        &self,
        logits: &mut [f32],
        context: &ProcessorContext,
        count: usize,
    ) -> anyhow::Result<usize> {
        self.process(logits, context)?;

        Ok(if self.top_k > 0 {
            count.min(self.top_k)
        } else {
            count
        })
    }

    fn device_sampling_transform(&self) -> Option<DeviceSamplingTransform> {
        Some(DeviceSamplingTransform::TopK(self.top_k))
    }

    fn process(&self, logits: &mut [f32], _context: &ProcessorContext) -> anyhow::Result<()> {
        let Some(mut boundary) = top_k_boundary(logits, self.top_k) else {
            return Ok(());
        };

        for logit in logits.iter_mut() {
            if !boundary.keeps(*logit) {
                *logit = f32::NEG_INFINITY;
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "top_k"
    }
}

/// Fused top-k then top-p.
///
/// Both are configured together in the common case, and running them as two
/// independent processors wastes most of the work. `TopKProcessor` prunes by
/// writing `-inf` in place while keeping the full vocabulary array, so
/// `TopPProcessor` then reads all of it -- on a 151,936-entry vocabulary with
/// `top_k = 40`, that is 151,896 entries scanned to learn they are already
/// excluded, twice (once to find the maximum, once to build the weights).
///
/// Fusing them keeps the k survivors from the top-k step and runs the nucleus
/// search on that set alone, which turns the top-p half from O(V) into O(k).
/// Measured at ~195 us versus ~0.5 us per token for the sizes above.
///
/// Behaviour is identical to running the two processors in sequence: the same
/// tokens survive, and the nucleus boundary uses the same f64 accumulation.
pub struct TopKTopPProcessor {
    pub top_k: usize,
    pub top_p: f32,
}

impl LogitProcessor for TopKTopPProcessor {
    fn process_with_candidates(
        &self,
        logits: &mut [f32],
        context: &ProcessorContext,
        candidates: &mut SamplingCandidates,
    ) -> anyhow::Result<()> {
        candidates.top_k(logits, self.top_k);
        let compact = self.top_p < 1.0
            && logits
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max)
                .is_finite();
        self.process(logits, context)?;
        if compact {
            candidates.retain(|id| logits[id].is_finite());
        }
        Ok(())
    }

    fn preserves_argmax(&self) -> bool {
        true
    }

    fn process_with_candidate_count(
        &self,
        logits: &mut [f32],
        context: &ProcessorContext,
        count: usize,
    ) -> anyhow::Result<usize> {
        self.process(logits, context)?;
        let live = logits.iter().filter(|value| value.is_finite()).count();
        let limited = if self.top_k > 0 {
            count.min(self.top_k)
        } else {
            count
        };
        Ok(if self.top_p < 1.0 && live > 0 {
            live.min(limited)
        } else {
            limited
        })
    }

    fn device_sampling_transform(&self) -> Option<DeviceSamplingTransform> {
        Some(DeviceSamplingTransform::TopKTopP {
            top_k: self.top_k,
            top_p: self.top_p,
        })
    }

    fn process(&self, logits: &mut [f32], _context: &ProcessorContext) -> anyhow::Result<()> {
        let Some(mut boundary) = top_k_boundary(logits, self.top_k) else {
            // No top-k pruning applies, so this degenerates to plain top-p.
            TopPProcessor { top_p: self.top_p }.process(logits, _context)?;
            return Ok(());
        };

        // Keep exactly k ranked candidates, resolving threshold ties by token
        // id just like the independent top-k stage. The shared boundary carries
        // the remaining tie quota, so fusion cannot admit extra tied tokens.
        let mut survivors: Vec<(usize, f32)> = Vec::with_capacity(self.top_k);
        let mut maximum_logit = f32::NEG_INFINITY;
        for (index, logit) in logits.iter_mut().enumerate() {
            if !boundary.keeps(*logit) {
                *logit = f32::NEG_INFINITY;
                continue;
            }
            if *logit > maximum_logit {
                maximum_logit = *logit;
            }
            survivors.push((index, *logit));
        }

        if !self.top_p.is_finite() || self.top_p >= 1.0 || survivors.is_empty() {
            return Ok(());
        }
        if !maximum_logit.is_finite() {
            return Ok(());
        }

        // Softmax over the survivors only.
        let mut total = 0.0_f32;
        for (_, value) in &mut survivors {
            let weight = (*value - maximum_logit).exp();
            *value = weight;
            total += weight;
        }
        if !total.is_finite() || total <= 0.0 {
            return Ok(());
        }
        let inverse_total = 1.0_f32 / total;
        let mut zero_probability: Vec<usize> = Vec::new();
        let mut probabilities: Vec<(usize, f32)> = Vec::with_capacity(survivors.len());
        for (index, weight) in survivors {
            let probability = weight * inverse_total;
            if probability > 0.0 {
                probabilities.push((index, probability));
            } else {
                zero_probability.push(index);
            }
        }

        for index in zero_probability {
            logits[index] = f32::NEG_INFINITY;
        }
        if probabilities.is_empty() {
            return Ok(());
        }

        let cutoff = self.top_p.max(0.0);
        let keep_count = top_p_keep_count(&mut probabilities, cutoff);
        for &(index, _) in probabilities.iter().skip(keep_count) {
            logits[index] = f32::NEG_INFINITY;
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "top_k_top_p"
    }
}

pub struct TopPProcessor {
    pub top_p: f32,
}

impl LogitProcessor for TopPProcessor {
    fn process_with_candidates(
        &self,
        logits: &mut [f32],
        context: &ProcessorContext,
        candidates: &mut SamplingCandidates,
    ) -> anyhow::Result<()> {
        let compact = self.top_p < 1.0
            && logits
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max)
                .is_finite();
        self.process(logits, context)?;
        if compact {
            candidates.retain(|id| logits[id].is_finite());
        }
        Ok(())
    }

    fn preserves_argmax(&self) -> bool {
        true
    }

    fn process_with_candidate_count(
        &self,
        logits: &mut [f32],
        context: &ProcessorContext,
        count: usize,
    ) -> anyhow::Result<usize> {
        self.process(logits, context)?;
        let live = logits.iter().filter(|value| value.is_finite()).count();
        Ok(if self.top_p < 1.0 {
            if live > 0 { live.min(count) } else { count }
        } else {
            count
        })
    }

    fn device_sampling_transform(&self) -> Option<DeviceSamplingTransform> {
        Some(DeviceSamplingTransform::TopP(self.top_p))
    }

    fn process(&self, logits: &mut [f32], _context: &ProcessorContext) -> anyhow::Result<()> {
        if !self.top_p.is_finite() || self.top_p >= 1.0 || logits.is_empty() {
            return Ok(());
        }

        let Some((mut probabilities, zero_probability_indices)) =
            nonzero_softmax_probabilities(logits)
        else {
            return Ok(());
        };
        if probabilities.is_empty() {
            return Ok(());
        }

        let cutoff = self.top_p.max(0.0);
        let keep_count = top_p_keep_count(&mut probabilities, cutoff);

        for idx in zero_probability_indices {
            logits[idx] = f32::NEG_INFINITY;
        }
        for &(idx, _) in probabilities.iter().skip(keep_count) {
            logits[idx] = f32::NEG_INFINITY;
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "top_p"
    }
}

pub struct MinPProcessor {
    pub min_p: f32,
}

impl LogitProcessor for MinPProcessor {
    fn process_with_candidates(
        &self,
        logits: &mut [f32],
        context: &ProcessorContext,
        candidates: &mut SamplingCandidates,
    ) -> anyhow::Result<()> {
        let compact = self.min_p > 0.0
            && logits
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max)
                .is_finite();
        self.process(logits, context)?;
        if compact {
            candidates.retain(|id| logits[id].is_finite());
        }
        Ok(())
    }

    fn preserves_argmax(&self) -> bool {
        true
    }

    fn process_with_candidate_count(
        &self,
        logits: &mut [f32],
        context: &ProcessorContext,
        count: usize,
    ) -> anyhow::Result<usize> {
        self.process(logits, context)?;
        let live = logits.iter().filter(|value| value.is_finite()).count();
        Ok(if self.min_p > 0.0 {
            if live > 0 { live.min(count) } else { count }
        } else {
            count
        })
    }

    fn device_sampling_transform(&self) -> Option<DeviceSamplingTransform> {
        Some(DeviceSamplingTransform::MinP(self.min_p))
    }

    fn process(&self, logits: &mut [f32], _context: &ProcessorContext) -> anyhow::Result<()> {
        if !self.min_p.is_finite() || self.min_p <= 0.0 || logits.is_empty() {
            return Ok(());
        }

        let Some(probabilities) = softmax_probabilities(logits) else {
            return Ok(());
        };

        let top_prob = probabilities.iter().copied().fold(0.0, f32::max);
        let threshold = self.min_p.min(1.0) * top_prob;
        for (logit, prob) in logits.iter_mut().zip(probabilities) {
            if !prob.is_finite() || prob < threshold {
                *logit = f32::NEG_INFINITY;
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "min_p"
    }
}

/// Top-A sampling keeps tokens with `p >= top_a * p_max^2`.
pub struct TopAProcessor {
    pub top_a: f32,
}

impl LogitProcessor for TopAProcessor {
    fn process_with_candidates(
        &self,
        logits: &mut [f32],
        context: &ProcessorContext,
        candidates: &mut SamplingCandidates,
    ) -> anyhow::Result<()> {
        let compact = self.top_a > 0.0
            && logits
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max)
                .is_finite();
        self.process(logits, context)?;
        if compact {
            candidates.retain(|id| logits[id].is_finite());
        }
        Ok(())
    }

    fn preserves_argmax(&self) -> bool {
        true
    }

    fn process_with_candidate_count(
        &self,
        logits: &mut [f32],
        context: &ProcessorContext,
        count: usize,
    ) -> anyhow::Result<usize> {
        self.process(logits, context)?;
        let live = logits.iter().filter(|value| value.is_finite()).count();
        Ok(if self.top_a > 0.0 {
            if live > 0 { live.min(count) } else { count }
        } else {
            count
        })
    }

    fn process(&self, logits: &mut [f32], _context: &ProcessorContext) -> anyhow::Result<()> {
        if !self.top_a.is_finite() || self.top_a <= 0.0 || logits.is_empty() {
            return Ok(());
        }
        let Some(probabilities) = softmax_probabilities(logits) else {
            return Ok(());
        };
        let maximum_probability = probabilities.iter().copied().fold(0.0, f32::max);
        let threshold = self.top_a.min(1.0) * maximum_probability * maximum_probability;
        for (logit, probability) in logits.iter_mut().zip(probabilities) {
            if probability < threshold {
                *logit = f32::NEG_INFINITY;
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "top_a"
    }
}

/// Locally typical sampling ordered by distance from the distribution entropy.
pub struct TypicalPProcessor {
    pub typical_p: f32,
}

impl LogitProcessor for TypicalPProcessor {
    fn process_with_candidates(
        &self,
        logits: &mut [f32],
        context: &ProcessorContext,
        candidates: &mut SamplingCandidates,
    ) -> anyhow::Result<()> {
        let compact = self.typical_p < 1.0
            && logits
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max)
                .is_finite();
        self.process(logits, context)?;
        if compact {
            candidates.retain(|id| logits[id].is_finite());
        }
        Ok(())
    }

    fn process_with_candidate_count(
        &self,
        logits: &mut [f32],
        context: &ProcessorContext,
        count: usize,
    ) -> anyhow::Result<usize> {
        self.process(logits, context)?;
        let live = logits.iter().filter(|value| value.is_finite()).count();
        Ok(if self.typical_p < 1.0 {
            if live > 0 { live.min(count) } else { count }
        } else {
            count
        })
    }

    fn process(&self, logits: &mut [f32], _context: &ProcessorContext) -> anyhow::Result<()> {
        if !self.typical_p.is_finite() || self.typical_p >= 1.0 || logits.is_empty() {
            return Ok(());
        }
        let Some(probabilities) = softmax_probabilities(logits) else {
            return Ok(());
        };
        let entropy: f32 = probabilities
            .iter()
            .copied()
            .filter(|probability| *probability > 0.0)
            .map(|probability| -probability * probability.ln())
            .sum();
        let mut ranked: Vec<_> = probabilities
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, probability)| *probability > 0.0)
            .map(|(index, probability)| (index, probability, (-probability.ln() - entropy).abs()))
            .collect();
        ranked.sort_unstable_by(|left, right| {
            left.2
                .total_cmp(&right.2)
                .then_with(|| right.1.total_cmp(&left.1))
                .then_with(|| left.0.cmp(&right.0))
        });

        let cutoff = self.typical_p.max(0.0);
        let mut cumulative = 0.0;
        let mut keep_count = 0;
        for &(_, probability, _) in &ranked {
            keep_count += 1;
            cumulative += probability;
            if cumulative >= cutoff {
                break;
            }
        }
        for &(index, _, _) in ranked.iter().skip(keep_count) {
            logits[index] = f32::NEG_INFINITY;
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "typical_p"
    }
}

#[derive(Debug)]
struct MirostatPreviousStep {
    generated_length: usize,
    probabilities: Vec<f32>,
}

#[derive(Debug)]
struct MirostatState {
    mu: f32,
    previous_step: Option<MirostatPreviousStep>,
}

/// Mirostat v1/v2 adaptive surprise controller.
///
/// The processor applies the current `mu` cutoff before sampling. On the next
/// call it observes the token appended to `generated_tokens`, measures that
/// token's surprise under the prior full distribution, and applies
/// `mu -= eta * (surprise - tau)`.
pub struct MirostatProcessor {
    pub tau: f32,
    pub eta: f32,
    pub version: MirostatVersion,
    state: Mutex<MirostatState>,
}

impl MirostatProcessor {
    pub fn new(tau: f32, eta: f32, version: MirostatVersion) -> Self {
        Self {
            tau,
            eta,
            version,
            state: Mutex::new(MirostatState {
                mu: 2.0 * tau,
                previous_step: None,
            }),
        }
    }

    #[cfg(test)]
    fn current_mu(&self) -> f32 {
        self.state.lock().expect("mirostat state").mu
    }

    fn v1_keep_count(probabilities: &[f32], mu: f32) -> usize {
        let ranked: Vec<_> = ranked_probabilities(probabilities)
            .into_iter()
            .filter(|(_, probability)| *probability > 0.0)
            .collect();
        let estimate_count = ranked.len().min(100);
        if estimate_count < 2 {
            return ranked.len().min(1);
        }

        let mut numerator = 0.0_f32;
        let mut denominator = 0.0_f32;
        for rank in 0..estimate_count - 1 {
            let probability = ranked[rank].1;
            let next_probability = ranked[rank + 1].1;
            if probability <= 0.0 || next_probability <= 0.0 {
                break;
            }
            let rank_ratio = ((rank + 2) as f32 / (rank + 1) as f32).ln();
            let probability_ratio = (probability / next_probability).ln();
            numerator += rank_ratio * probability_ratio;
            denominator += rank_ratio * rank_ratio;
        }
        let slope = numerator / denominator;
        let epsilon = slope - 1.0;
        let vocabulary = ranked.len() as f32;
        let keep =
            ((epsilon * 2.0_f32.powf(mu)) / (1.0 - vocabulary.powf(-epsilon))).powf(1.0 / slope);
        if !keep.is_finite() || slope <= 1.0 || epsilon <= 0.0 {
            return mirostat_v2_keep_count(&ranked, mu);
        }
        (keep.round() as usize).clamp(1, ranked.len())
    }
}

impl LogitProcessor for MirostatProcessor {
    fn process(&self, logits: &mut [f32], context: &ProcessorContext) -> anyhow::Result<()> {
        if !self.tau.is_finite()
            || self.tau <= 0.0
            || !self.eta.is_finite()
            || self.eta <= 0.0
            || logits.is_empty()
        {
            return Ok(());
        }
        let Some(probabilities) = softmax_probabilities(logits) else {
            return Ok(());
        };
        let mut state = self.state.lock().map_err(|_| {
            anyhow::anyhow!("Mirostat request state is poisoned; restart this generation")
        })?;

        if let Some(previous) = state.previous_step.take() {
            if context.generated_tokens.len() < previous.generated_length {
                state.mu = 2.0 * self.tau;
            } else if let Some(&token_id) = context.generated_tokens.get(previous.generated_length)
                && let Some(&probability) = previous.probabilities.get(token_id as usize)
                && probability > 0.0
            {
                let surprise = -probability.log2();
                state.mu -= self.eta * (surprise - self.tau);
            }
        }

        let ranked = ranked_probabilities(&probabilities);
        let keep_count = match self.version {
            MirostatVersion::V1 => Self::v1_keep_count(&probabilities, state.mu),
            MirostatVersion::V2 => mirostat_v2_keep_count(&ranked, state.mu),
        };
        for &(index, _) in ranked.iter().skip(keep_count) {
            logits[index] = f32::NEG_INFINITY;
        }
        state.previous_step = Some(MirostatPreviousStep {
            generated_length: context.generated_tokens.len(),
            probabilities,
        });
        Ok(())
    }

    fn name(&self) -> &str {
        match self.version {
            MirostatVersion::V1 => "mirostat_v1",
            MirostatVersion::V2 => "mirostat_v2",
        }
    }
}

fn mirostat_v2_keep_count(ranked: &[(usize, f32)], mu: f32) -> usize {
    ranked
        .iter()
        .position(|(_, probability)| *probability <= 0.0 || -probability.log2() > mu)
        .unwrap_or(ranked.len())
        .max(1)
}

/// Immutable byte-prefix and declared ending roles shared by repeated stages.
pub(crate) struct InfillContract {
    pieces: Vec<Vec<u8>>,
    positions: HashMap<usize, usize>,
    prefixes: Vec<std::ops::Range<usize>>,
    endings: std::collections::BTreeSet<usize>,
    fallback: usize,
}
impl InfillContract {
    pub(crate) fn new(
        tokenizer: &onnx_genai_ort::Tokenizer,
        endings: Vec<TokenId>,
        fallback: TokenId,
    ) -> anyhow::Result<Self> {
        let mut ids: Vec<_> = tokenizer.inner().get_vocab(true).into_values().collect();
        ids.sort_unstable();
        ids.dedup();
        let mut bytes = crate::tokenization::token_pieces(tokenizer, &ids).context(
            "infill requires lossless tokenizer byte pieces; repair the tokenizer decoder",
        )?;
        let specials = tokenizer.inner().get_added_tokens_decoder();
        for (id, piece) in ids.iter().zip(&mut bytes) {
            if specials.get(id).is_some_and(|token| token.special) {
                piece.clear();
            }
        }
        Self::from_pieces(ids, bytes, endings, fallback)
    }
    fn from_pieces(
        ids: Vec<TokenId>,
        bytes: Vec<Vec<u8>>,
        endings: Vec<TokenId>,
        fallback: TokenId,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            ids.len() == bytes.len(),
            "infill tokenizer byte pieces must match the vocabulary ids"
        );
        anyhow::ensure!(
            !endings.is_empty() && endings.contains(&fallback),
            "infill requires a declared end-of-turn/EOS fallback in the effective EOS set; declare package.tokenizer.special_tokens.eos_token_id"
        );
        let mut entries: Vec<_> = ids.into_iter().zip(bytes).collect();
        entries.sort_unstable_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        let positions: HashMap<_, _> = entries
            .iter()
            .enumerate()
            .map(|(position, (id, _))| (*id as usize, position))
            .collect();
        anyhow::ensure!(
            positions.len() == entries.len(),
            "infill tokenizer byte pieces repeat a token id; repair the tokenizer vocabulary contract"
        );
        for id in &endings {
            anyhow::ensure!(
                positions.contains_key(&(*id as usize)),
                "infill ending token {id} is absent from the tokenizer vocabulary; repair the package token facts"
            );
        }
        let pieces: Vec<_> = entries.into_iter().map(|(_, bytes)| bytes).collect();
        let prefixes = pieces
            .iter()
            .map(|piece| {
                let start = pieces.partition_point(|other| other.as_slice() < piece.as_slice());
                let end = pieces.partition_point(|other| {
                    other.as_slice() < piece.as_slice() || other.starts_with(piece)
                });
                start..end
            })
            .collect();
        Ok(Self {
            pieces,
            positions,
            prefixes,
            endings: endings.into_iter().map(|id| id as usize).collect(),
            fallback: fallback as usize,
        })
    }
}

/// Range minimum over live candidate ranks in lexicographic byte-piece order.
/// Each merge removes a candidate, so indexed prefix traversal performs at most
/// O(V) merges and O(V log V) work, retaining upstream's probability-rank order.
struct LivePrefixRanks {
    base: usize,
    nodes: Vec<usize>,
}
impl LivePrefixRanks {
    fn new(size: usize) -> anyhow::Result<Self> {
        let base = size
            .max(1)
            .checked_next_power_of_two()
            .context("infill vocabulary is too large for the host prefix index")?;
        let size = base
            .checked_mul(2)
            .context("infill prefix index size overflows the host address space")?;
        Ok(Self {
            base,
            nodes: vec![usize::MAX; size],
        })
    }
    fn set(&mut self, position: usize, rank: usize) {
        let mut at = self.base + position;
        self.nodes[at] = rank;
        while at > 1 {
            at /= 2;
            self.nodes[at] = self.nodes[2 * at].min(self.nodes[2 * at + 1]);
        }
    }
    fn first(&self, range: std::ops::Range<usize>) -> Option<usize> {
        let mut left = self.base + range.start;
        let mut right = self.base + range.end;
        let mut best = usize::MAX;
        while left < right {
            if left % 2 == 1 {
                best = best.min(self.nodes[left]);
                left += 1;
            }
            if right % 2 == 1 {
                right -= 1;
                best = best.min(self.nodes[right]);
            }
            left /= 2;
            right /= 2;
        }
        (best != usize::MAX).then_some(best)
    }
}

pub(crate) struct InfillProcessor {
    pub(crate) contract: std::sync::Arc<InfillContract>,
}
impl LogitProcessor for InfillProcessor {
    fn needs_candidate_identity(&self) -> bool {
        true
    }
    fn process(&self, logits: &mut [f32], context: &ProcessorContext) -> anyhow::Result<()> {
        let mut candidates = SamplingCandidates::new(logits);
        self.process_with_candidates(logits, context, &mut candidates)
    }
    fn process_with_candidate_count(
        &self,
        logits: &mut [f32],
        context: &ProcessorContext,
        count: usize,
    ) -> anyhow::Result<usize> {
        anyhow::ensure!(
            count == logits.len(),
            "infill requires candidate identities after truncation; execute it through ProcessorChain"
        );
        let mut candidates = SamplingCandidates::new(logits);
        self.process_with_candidates(logits, context, &mut candidates)?;
        Ok(candidates.len())
    }
    fn process_with_candidates(
        &self,
        logits: &mut [f32],
        _context: &ProcessorContext,
        candidates: &mut SamplingCandidates,
    ) -> anyhow::Result<()> {
        let contract = &self.contract;
        let mut ranked = candidates.indices.clone();
        ranked.sort_unstable_by(|&a, &b| compare_candidate_logits(a, b, logits));
        let maximum=ranked.first().map(|&id|logits[id]).filter(|v|v.is_finite()).context("infill has no finite candidate to normalize; reduce penalties/token bans or inspect model logits")?;
        let mut weights = vec![0.0_f32; logits.len()];
        let mut total = 0.0_f32;
        for &id in &ranked {
            let weight = if logits[id].is_nan() {
                0.0
            } else {
                (logits[id] - maximum).exp()
            };
            weights[id] = weight;
            total += weight;
        }
        anyhow::ensure!(
            total.is_finite() && total > 0.0,
            "infill candidate normalization failed; provide finite model logits and at least one available token"
        );
        let mut text_mass = 0.0;
        let mut ending_mass = 0.0;
        for &id in &ranked {
            weights[id] /= total;
            if contract.endings.contains(&id) {
                ending_mass += weights[id];
            } else {
                text_mass += weights[id];
            }
        }
        if 3.0 * ending_mass * ranked.len() as f32 > text_mass {
            candidates
                .indices
                .retain(|id| contract.endings.contains(id));
            for &id in &ranked {
                if !contract.endings.contains(&id) {
                    logits[id] = f32::NEG_INFINITY;
                }
            }
            return Ok(());
        }
        let mut tree = LivePrefixRanks::new(contract.pieces.len())?;
        let mut leaves = vec![None; ranked.len()];
        for (rank, &id) in ranked.iter().enumerate() {
            if logits[id].is_nan() || logits[id] == f32::NEG_INFINITY {
                continue;
            }
            let leaf=*contract.positions.get(&id).with_context(||format!("infill candidate token {id} has no tokenizer byte piece; repair the tokenizer/model vocabulary contract"))?;
            leaves[rank] = Some(leaf);
            tree.set(leaf, rank);
        }
        for (rank, &id) in ranked.iter().enumerate() {
            if logits[id] == f32::NEG_INFINITY || logits[id].is_nan() {
                continue;
            }
            let leaf = leaves[rank].context("infill live candidate lost its byte-piece index")?;
            if contract.pieces[leaf].is_empty() {
                continue;
            }
            tree.set(leaf, usize::MAX);
            while let Some(other_rank) = tree.first(contract.prefixes[leaf].clone()) {
                let other = ranked[other_rank];
                if weights[other] > weights[id] {
                    weights[other] += weights[id];
                    weights[id] = 0.0;
                    logits[id] = f32::NEG_INFINITY;
                    break;
                } else {
                    weights[id] += weights[other];
                    weights[other] = 0.0;
                    logits[other] = f32::NEG_INFINITY;
                    tree.set(
                        leaves[other_rank]
                            .context("infill merged candidate lost its byte-piece index")?,
                        usize::MAX,
                    );
                }
            }
            if logits[id] != f32::NEG_INFINITY {
                tree.set(leaf, rank);
            }
        }
        let mut non_endings = 0;
        let mut retained_mass = 0.0;
        ranked.retain(|&id| {
            let end = contract.endings.contains(&id);
            let keep = end || weights[id] >= 0.2;
            if keep {
                retained_mass += weights[id];
                if !end {
                    non_endings += 1;
                }
            } else {
                logits[id] = f32::NEG_INFINITY;
            }
            keep
        });
        if non_endings == 0 {
            anyhow::ensure!(
                contract.fallback < logits.len(),
                "infill end token {} is outside model logits; repair the tokenizer/model token contract",
                contract.fallback
            );
            anyhow::ensure!(
                candidates.hard_allowed.get(contract.fallback) == Some(&true),
                "infill needs end token {} but its input/model logit or a mandatory token/grammar mask excludes it; permit a declared ending token or relax infill filtering",
                contract.fallback
            );
            logits.fill(f32::NEG_INFINITY);
            logits[contract.fallback] = 1.0;
            candidates.indices = vec![contract.fallback];
            return Ok(());
        }
        let threshold = 1.0 / (non_endings + 1) as f32;
        ranked.retain(|&id| {
            weights[id] /= retained_mass;
            let keep = contract.endings.contains(&id) || weights[id] >= threshold;
            if !keep {
                logits[id] = f32::NEG_INFINITY;
            }
            keep
        });
        candidates.indices = ranked;
        Ok(())
    }
    fn name(&self) -> &str {
        "infill"
    }
}

/// Request-local adaptive probability transform. The ordinary terminal sampler
/// supplies the draw; selected-token feedback records original probabilities.
pub(crate) struct AdaptivePProcessor {
    config: crate::AdaptivePConfig,
    state: Mutex<AdaptivePState>,
}
#[derive(Clone, Copy)]
struct AdaptivePStats {
    weighted_sum: f32,
    total_weight: f32,
}
struct AdaptivePSelection {
    token: TokenId,
    before: AdaptivePStats,
}
struct AdaptivePPending {
    generated_len: usize,
    original_probabilities: Vec<f32>,
}
struct AdaptivePState {
    stats: AdaptivePStats,
    selections: Vec<AdaptivePSelection>,
    pending: Option<AdaptivePPending>,
}
impl AdaptivePProcessor {
    pub(crate) fn new(config: crate::AdaptivePConfig) -> anyhow::Result<Self> {
        config.validate()?;
        anyhow::ensure!(
            config.target >= 0.0,
            "adaptive_p processor requires a non-negative adaptive_target; omit the processor when adaptation is disabled"
        );
        Ok(Self {
            config,
            state: Mutex::new(AdaptivePState {
                stats: AdaptivePStats {
                    weighted_sum: config.target / (1.0 - config.decay),
                    total_weight: 1.0 / (1.0 - config.decay),
                },
                selections: Vec::new(),
                pending: None,
            }),
        })
    }
}
impl LogitProcessor for AdaptivePProcessor {
    fn process(&self, logits: &mut [f32], context: &ProcessorContext) -> anyhow::Result<()> {
        let mut state = self.state.lock().map_err(|_| {
            anyhow::anyhow!("adaptive_p request state is poisoned; restart this generation")
        })?;
        let common = state
            .selections
            .iter()
            .zip(&context.generated_tokens)
            .take_while(|(selection, token)| selection.token == **token)
            .count();
        if common < state.selections.len() {
            state.stats = state.selections[common].before;
            state.selections.truncate(common);
        }
        anyhow::ensure!(
            common == context.generated_tokens.len(),
            "adaptive_p history contains tokens without selection feedback at offset {common}; restart generation or use the engine's token-selection pipeline"
        );
        state.pending = None;
        let probabilities = softmax_probabilities(logits).ok_or_else(|| anyhow::anyhow!(
            "adaptive_p could not normalize model logits; leave a finite unmasked candidate by reducing penalties/token bans, or inspect non-finite model outputs"
        ))?;
        let target = (2.0 * self.config.target
            - state.stats.weighted_sum / state.stats.total_weight)
            .clamp(0.0, 1.0);
        for (logit, probability) in logits.iter_mut().zip(&probabilities) {
            if logit.is_nan() || *logit == f32::NEG_INFINITY {
                *logit = f32::NEG_INFINITY;
                continue;
            }
            let distance = ((*probability - target) * (1.0 / 0.3)).abs();
            *logit = 5.0 - 10.0 * distance * distance / (1.0 + distance);
        }
        state.pending = Some(AdaptivePPending {
            generated_len: context.generated_tokens.len(),
            original_probabilities: probabilities,
        });
        Ok(())
    }
    fn needs_selection_feedback(&self) -> bool {
        true
    }
    fn on_selected(&self, token: TokenId, context: &ProcessorContext) -> anyhow::Result<()> {
        let mut state = self.state.lock().map_err(|_| {
            anyhow::anyhow!("adaptive_p request state is poisoned; restart this generation")
        })?;
        let pending = state.pending.take().ok_or_else(|| anyhow::anyhow!("adaptive_p selection has no pending probability distribution; process logits before staging a token"))?;
        anyhow::ensure!(
            pending.generated_len == context.generated_tokens.len()
                && state.selections.len() == pending.generated_len
                && state
                    .selections
                    .iter()
                    .zip(&context.generated_tokens)
                    .all(|(selection, token)| selection.token == *token),
            "adaptive_p selection history changed after logits were processed; restart this generation"
        );
        let probability = *pending.original_probabilities.get(token as usize).ok_or_else(|| anyhow::anyhow!(
            "adaptive_p selected token {token} is outside the processed vocabulary; select a token from the model logits"
        ))?;
        let before = state.stats;
        state.stats = AdaptivePStats {
            weighted_sum: probability + self.config.decay * before.weighted_sum,
            total_weight: 1.0 + self.config.decay * before.total_weight,
        };
        state.selections.push(AdaptivePSelection { token, before });
        Ok(())
    }
    fn name(&self) -> &str {
        "adaptive_p"
    }
}

/// XTC randomly removes all but the least-probable token among top choices.
pub struct XtcProcessor {
    pub probability: f32,
    pub threshold: f32,
    rng: Mutex<StdRng>,
}

impl XtcProcessor {
    pub fn new(probability: f32, threshold: f32, seed: Option<u64>) -> Self {
        let rng = seed.map_or_else(StdRng::from_os_rng, StdRng::seed_from_u64);
        Self {
            probability,
            threshold,
            rng: Mutex::new(rng),
        }
    }
}

impl LogitProcessor for XtcProcessor {
    fn process_with_candidates(
        &self,
        logits: &mut [f32],
        context: &ProcessorContext,
        candidates: &mut SamplingCandidates,
    ) -> anyhow::Result<()> {
        let masked: Vec<bool> = logits
            .iter()
            .map(|v| *v == f32::NEG_INFINITY || v.is_nan())
            .collect();
        self.process(logits, context)?;
        candidates.retain(|id| masked[id] || logits[id] != f32::NEG_INFINITY);
        Ok(())
    }

    fn process_with_candidate_count(
        &self,
        logits: &mut [f32],
        context: &ProcessorContext,
        count: usize,
    ) -> anyhow::Result<usize> {
        let before = logits.iter().filter(|value| value.is_finite()).count();
        self.process(logits, context)?;
        let after = logits.iter().filter(|value| value.is_finite()).count();
        Ok(count.saturating_sub(before.saturating_sub(after)))
    }

    fn process(&self, logits: &mut [f32], _context: &ProcessorContext) -> anyhow::Result<()> {
        if !self.probability.is_finite()
            || self.probability <= 0.0
            || !self.threshold.is_finite()
            || logits.is_empty()
        {
            return Ok(());
        }
        let should_apply = match self.rng.lock() {
            Ok(mut rng) => rng.random::<f32>() < self.probability.min(1.0),
            Err(_) => false,
        };
        if !should_apply {
            return Ok(());
        }
        let Some(probabilities) = softmax_probabilities(logits) else {
            return Ok(());
        };
        let candidates: Vec<_> = probabilities
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, probability)| *probability > 0.0 && *probability > self.threshold.max(0.0))
            .collect();
        if candidates.len() < 2 {
            return Ok(());
        }
        let keep_index = candidates
            .iter()
            .min_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| left.0.cmp(&right.0))
            })
            .map(|(index, _)| *index)
            .expect("XTC candidates are non-empty");
        for (index, _) in candidates {
            if index != keep_index {
                logits[index] = f32::NEG_INFINITY;
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "xtc"
    }
}

fn softmax_probabilities(logits: &[f32]) -> Option<Vec<f32>> {
    let maximum_logit = logits
        .iter()
        .copied()
        .filter(|value| !value.is_nan())
        .fold(f32::NEG_INFINITY, f32::max);
    if !maximum_logit.is_finite() {
        return None;
    }
    let weights: Vec<_> = logits
        .iter()
        .map(|&logit| {
            if logit.is_nan() {
                0.0
            } else {
                (logit - maximum_logit).exp()
            }
        })
        .collect();
    let total: f32 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return None;
    }
    Some(weights.into_iter().map(|weight| weight / total).collect())
}

/// Surviving `(index, probability)` pairs together with the indices whose
/// probability is forced to zero. Named to keep the sampler helpers readable
/// and satisfy `clippy::type_complexity`.
type NonzeroProbabilities = (Vec<(usize, f32)>, Vec<usize>);

/// Softmax the finite logits, returning the surviving `(index, probability)`
/// pairs and the indices forced to zero.
///
/// This runs once per generated token, so its allocation behaviour is part of
/// the decode cost rather than a detail. Two things matter here:
///
/// - **No staging vector.** An earlier version collected `(index, logit)` pairs
///   just to find the maximum, then consumed that vector to build the weights.
///   On a 151k-entry vocabulary that is an extra 1.2 MB allocation, filled and
///   thrown away, every token. The maximum is now found in a scan that keeps
///   nothing.
/// - **Capacity is known.** `weights` can hold at most one entry per logit, so
///   it is reserved once instead of growing through ~18 reallocations. The
///   zero list usually stays empty -- most vocabularies have no NaN and no
///   `-inf` at this point -- so it does not allocate at all unless it must.
///
/// Measured on this shape (one dominant mode, long tail), this is ~2.7x faster
/// at a 151k vocabulary and ~2.2x at 128k than the staging-vector version.
fn nonzero_softmax_probabilities(logits: &[f32]) -> Option<NonzeroProbabilities> {
    let mut maximum_logit = f32::NEG_INFINITY;
    let mut has_zero_probability = false;
    for &logit in logits {
        if logit.is_nan() {
            has_zero_probability = true;
        } else if logit != f32::NEG_INFINITY && logit > maximum_logit {
            maximum_logit = logit;
        }
    }
    if !maximum_logit.is_finite() {
        return None;
    }

    let mut weights: Vec<(usize, f32)> = Vec::with_capacity(logits.len());
    // Only pay for this when the first pass proved something needs it. A logit
    // can also underflow to a zero weight below, which pushes here and grows
    // from empty -- rare enough that reserving for it would cost more than it
    // saves.
    let mut zero_probability_indices: Vec<usize> = if has_zero_probability {
        Vec::with_capacity(16)
    } else {
        Vec::new()
    };
    let mut total = 0.0_f32;
    for (index, &logit) in logits.iter().enumerate() {
        if logit.is_nan() {
            zero_probability_indices.push(index);
            continue;
        }
        if logit == f32::NEG_INFINITY {
            continue;
        }
        let weight = (logit - maximum_logit).exp();
        total += weight;
        if weight > 0.0 {
            weights.push((index, weight));
        } else {
            zero_probability_indices.push(index);
        }
    }
    if !total.is_finite() || total <= 0.0 {
        return None;
    }

    // One reciprocal and a multiply per entry rather than a divide per entry.
    let inverse_total = 1.0_f32 / total;
    for (_, weight) in &mut weights {
        *weight *= inverse_total;
    }
    Some((weights, zero_probability_indices))
}

/// Exactly-k boundary. Equal logits consume the quota in ascending token order.
struct TopKBoundary {
    threshold: f32,
    remaining_ties: usize,
}

impl TopKBoundary {
    fn keeps(&mut self, value: f32) -> bool {
        if value > self.threshold {
            true
        } else if value == self.threshold && self.remaining_ties > 0 {
            self.remaining_ties -= 1;
            true
        } else {
            false
        }
    }
}

fn top_k_boundary(logits: &[f32], top_k: usize) -> Option<TopKBoundary> {
    if top_k == 0 || top_k >= logits.len() {
        return None;
    }

    if top_k <= 256 {
        return top_k_boundary_bounded_heap(logits, top_k);
    }

    let mut candidates: Vec<f32> = logits
        .iter()
        .copied()
        .filter(|value| !value.is_nan())
        .collect();
    if candidates.is_empty() {
        return None;
    }

    let rank = top_k.saturating_sub(1).min(candidates.len() - 1);
    let (above, threshold, _) = candidates.select_nth_unstable_by(rank, |left, right| {
        right.partial_cmp(left).unwrap_or(std::cmp::Ordering::Equal)
    });
    Some(TopKBoundary {
        threshold: *threshold,
        remaining_ties: rank + 1 - above.iter().filter(|&&value| value > *threshold).count(),
    })
}

fn top_k_boundary_bounded_heap(logits: &[f32], top_k: usize) -> Option<TopKBoundary> {
    let mut heap = BinaryHeap::with_capacity(top_k);
    for value in logits.iter().copied().filter(|value| !value.is_nan()) {
        if heap.len() < top_k {
            heap.push(std::cmp::Reverse(TopKLogit(value)));
        } else if let Some(mut threshold) = heap.peek_mut()
            && value > threshold.0.0
        {
            *threshold = std::cmp::Reverse(TopKLogit(value));
        }
    }
    let threshold = heap.peek()?.0.0;
    Some(TopKBoundary {
        threshold,
        remaining_ties: heap.iter().filter(|value| value.0.0 == threshold).count(),
    })
}

fn compare_ranked_probability(left: &(usize, f32), right: &(usize, f32)) -> std::cmp::Ordering {
    right
        .1
        .total_cmp(&left.1)
        .then_with(|| left.0.cmp(&right.0))
}

/// Cumulative sum in strict descending order, the reference behaviour.
///
/// Accumulates in `f64` even though the probabilities are `f32`. Floating-point
/// addition is not associative, so a `f32` running total makes the boundary
/// sensitive to summation order -- which is what let the partitioned fast path
/// below disagree with this one by a token. Widening only the accumulator costs
/// four bytes on the stack and removes that entire class of difference.
fn sorted_top_p_keep_count(probabilities: &mut [(usize, f32)], cutoff: f64) -> usize {
    probabilities.sort_unstable_by(compare_ranked_probability);
    let mut cumulative = 0.0_f64;
    for (offset, &(_, probability)) in probabilities.iter().enumerate() {
        cumulative += f64::from(probability);
        if cumulative >= cutoff {
            return offset + 1;
        }
    }
    probabilities.len()
}

fn top_p_keep_count(probabilities: &mut [(usize, f32)], cutoff: f32) -> usize {
    if probabilities.is_empty() {
        return 0;
    }
    if cutoff <= 0.0 {
        probabilities.select_nth_unstable_by(0, compare_ranked_probability);
        return 1;
    }
    let cutoff = f64::from(cutoff);
    if probabilities.len() <= 64 {
        return sorted_top_p_keep_count(probabilities, cutoff);
    }

    partition_top_p_keep_count(probabilities, cutoff)
}

/// Locate the nucleus boundary by halving instead of fully sorting.
///
/// Sums each half's mass and recurses into whichever half contains the cutoff,
/// so only the region around the boundary is ever ordered. The mass is
/// accumulated in `f64` for the reason given on [`sorted_top_p_keep_count`]:
/// with an `f32` total this produced a keep count one token away from the
/// strict descending sum on roughly 0.2% of realistic distributions, which
/// changed the sampled token for a narrow band of RNG values. In `f64` the two
/// agree exactly across a 120,000-case fuzz of random softmax distributions.
fn partition_top_p_keep_count(probabilities: &mut [(usize, f32)], cutoff: f64) -> usize {
    if probabilities.len() <= 64 {
        return sorted_top_p_keep_count(probabilities, cutoff);
    }

    let prefix_len = probabilities.len() / 2;
    probabilities.select_nth_unstable_by(prefix_len - 1, compare_ranked_probability);
    let prefix_mass: f64 = probabilities[..prefix_len]
        .iter()
        .map(|&(_, probability)| f64::from(probability))
        .sum();
    if prefix_mass >= cutoff {
        partition_top_p_keep_count(&mut probabilities[..prefix_len], cutoff)
    } else {
        prefix_len
            + partition_top_p_keep_count(&mut probabilities[prefix_len..], cutoff - prefix_mass)
    }
}

fn ranked_probabilities(probabilities: &[f32]) -> Vec<(usize, f32)> {
    let mut ranked: Vec<_> = probabilities.iter().copied().enumerate().collect();
    ranked.sort_unstable_by(compare_ranked_probability);
    ranked
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::{sample_categorical, sample_greedy};

    fn deterministic_logits(len: usize) -> Vec<f32> {
        let mut state = 0x5EBA_571A_CAFE_F00D_u64;
        (0..len)
            .map(|index| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                if index % 997 == 0 {
                    f32::NAN
                } else {
                    let bucket = (state >> 32) as u32;
                    (bucket as f32 / u32::MAX as f32) * 40.0 - 20.0
                }
            })
            .collect()
    }

    fn full_sort_top_k_threshold(logits: &[f32], top_k: usize) -> Option<f32> {
        if top_k == 0 || top_k >= logits.len() {
            return None;
        }
        let mut sorted: Vec<f32> = logits
            .iter()
            .copied()
            .filter(|value| !value.is_nan())
            .collect();
        if sorted.is_empty() {
            return None;
        }
        sorted.sort_unstable_by(|left, right| {
            right.partial_cmp(left).unwrap_or(std::cmp::Ordering::Equal)
        });
        Some(sorted[top_k.saturating_sub(1).min(sorted.len() - 1)])
    }

    fn full_sort_top_k(logits: &mut [f32], top_k: usize) {
        if top_k == 0 || top_k >= logits.len() {
            return;
        }
        let mut ranked: Vec<_> = logits
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, value)| !value.is_nan())
            .collect();
        ranked.sort_unstable_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap()
                .then_with(|| left.0.cmp(&right.0))
        });
        let mut kept = vec![false; logits.len()];
        for &(index, _) in ranked.iter().take(top_k) {
            kept[index] = true;
        }
        for (index, logit) in logits.iter_mut().enumerate() {
            if !kept[index] {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    fn full_sort_top_p(logits: &mut [f32], top_p: f32) {
        if !top_p.is_finite() || top_p >= 1.0 || logits.is_empty() {
            return;
        }
        let Some(probabilities) = softmax_probabilities(logits) else {
            return;
        };
        let probs = ranked_probabilities(&probabilities);
        let mut cumulative = 0.0;
        let mut keep_count = 0;
        let cutoff = top_p.max(0.0);
        for &(_, probability) in &probs {
            keep_count += 1;
            cumulative += probability;
            if cumulative >= cutoff {
                break;
            }
        }
        for &(index, _) in probs.iter().skip(keep_count) {
            logits[index] = f32::NEG_INFINITY;
        }
    }

    fn context(prompt_tokens: Vec<TokenId>, generated_tokens: Vec<TokenId>) -> ProcessorContext {
        ProcessorContext {
            prompt_tokens,
            generated_tokens,
            generated_text: String::new(),
            step: 0,
        }
    }

    fn infill_fixture(
        pieces: &[&[u8]],
        endings: Vec<TokenId>,
        fallback: TokenId,
    ) -> std::sync::Arc<InfillContract> {
        std::sync::Arc::new(
            InfillContract::from_pieces(
                (0..pieces.len() as u32).collect(),
                pieces.iter().map(|p| p.to_vec()).collect(),
                endings,
                fallback,
            )
            .unwrap(),
        )
    }

    // Literal small-input reference of upstream's rank-ordered double loop.
    fn reference_infill(
        contract: &InfillContract,
        logits: &mut [f32],
        candidates: &mut SamplingCandidates,
    ) -> anyhow::Result<()> {
        let mut ids = candidates.indices.clone();
        ids.sort_unstable_by(|&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        let max = logits[*ids.first().unwrap()];
        anyhow::ensure!(max.is_finite(), "unusable row");
        let mut p = vec![0.0; logits.len()];
        let mut sum = 0.0;
        for &id in &ids {
            p[id] = (logits[id] - max).exp();
            sum += p[id];
        }
        let mut text = 0.0;
        let mut end = 0.0;
        for &id in &ids {
            p[id] /= sum;
            if contract.endings.contains(&id) {
                end += p[id]
            } else {
                text += p[id]
            }
        }
        if 3.0 * end * ids.len() as f32 > text {
            ids.retain(|id| contract.endings.contains(id));
            for id in &candidates.indices {
                if !contract.endings.contains(id) {
                    logits[*id] = f32::NEG_INFINITY;
                }
            }
            candidates.indices = ids;
            return Ok(());
        }
        for &a in &ids {
            for &b in &ids {
                if logits[a] == f32::NEG_INFINITY {
                    break;
                }
                if a == b || logits[b] == f32::NEG_INFINITY {
                    continue;
                }
                let a_bytes = &contract.pieces[contract.positions[&a]];
                let b_bytes = &contract.pieces[contract.positions[&b]];
                if !a_bytes.is_empty() && b_bytes.starts_with(a_bytes) {
                    let (dst, src) = if p[b] > p[a] { (b, a) } else { (a, b) };
                    p[dst] += p[src];
                    p[src] = 0.0;
                    logits[src] = f32::NEG_INFINITY;
                }
            }
        }
        ids.retain(|id| {
            let keep = p[*id] >= 0.2 || contract.endings.contains(id);
            if !keep {
                logits[*id] = f32::NEG_INFINITY;
            }
            keep
        });
        let n = ids
            .iter()
            .filter(|id| !contract.endings.contains(id))
            .count();
        if n == 0 {
            anyhow::ensure!(
                candidates.hard_allowed[contract.fallback],
                "mandatory exclusion"
            );
            logits.fill(f32::NEG_INFINITY);
            logits[contract.fallback] = 1.0;
            candidates.indices = vec![contract.fallback];
            return Ok(());
        }
        let total: f32 = ids.iter().map(|id| p[*id]).sum();
        ids.retain(|id| {
            let keep = p[*id] / total >= 1.0 / (n + 1) as f32 || contract.endings.contains(id);
            if !keep {
                logits[*id] = f32::NEG_INFINITY;
            }
            keep
        });
        candidates.indices = ids;
        Ok(())
    }

    #[test]
    fn infill_end_role_does_not_guess_a_special_token_type() {
        let tokenizer = onnx_genai_ort::Tokenizer::from_file(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/tiny-llm/tokenizer.json"),
        )
        .unwrap();
        let contract = InfillContract::new(&tokenizer, vec![5], 5).unwrap();
        assert_eq!(contract.pieces[contract.positions[&5]], b"world");
    }

    #[test]
    fn infill_ordinary_ending_bytes_participate_in_prefix_merging() {
        let contract = infill_fixture(&[b"ab", b"a"], vec![1], 1);
        let mut logits = vec![0.99_f32.ln(), 0.01_f32.ln()];
        let mut set = SamplingCandidates::new(&logits);
        InfillProcessor { contract }
            .process_with_candidates(&mut logits, &ProcessorContext::default(), &mut set)
            .unwrap();
        assert_eq!(set.indices, [0, 1]);
        assert_eq!(
            logits[1],
            f32::NEG_INFINITY,
            "ordinary EOS prefix merges into the higher-probability continuation"
        );
        assert_eq!(logits[0], 0.99_f32.ln());
    }

    #[test]
    fn infill_identity_contracts_match_compacting_and_mask_only_counts() {
        fn filters() -> Vec<Box<dyn LogitProcessor>> {
            vec![
                Box::new(TopKProcessor { top_k: 2 }),
                Box::new(TopKProcessor { top_k: 0 }),
                Box::new(TopKTopPProcessor {
                    top_k: 2,
                    top_p: 0.7,
                }),
                Box::new(TopKTopPProcessor {
                    top_k: 0,
                    top_p: 0.7,
                }),
                Box::new(TopPProcessor { top_p: 0.8 }),
                Box::new(TopPProcessor { top_p: 1.0 }),
                Box::new(MinPProcessor { min_p: 0.2 }),
                Box::new(TopAProcessor { top_a: 0.7 }),
                Box::new(TypicalPProcessor { typical_p: 0.7 }),
                Box::new(XtcProcessor::new(1.0, 0.2, Some(11))),
            ]
        }
        for (counted, identified) in filters().into_iter().zip(filters()) {
            let original = vec![0.5_f32.ln(), 0.3_f32.ln(), 0.2_f32.ln(), f32::NEG_INFINITY];
            let mut old = original.clone();
            let mut new = original.clone();
            let expected = counted
                .process_with_candidate_count(&mut old, &ProcessorContext::default(), 4)
                .unwrap();
            let mut set = SamplingCandidates::new(&original);
            identified
                .process_with_candidates(&mut new, &ProcessorContext::default(), &mut set)
                .unwrap();
            assert_eq!(set.len(), expected, "{}", identified.name());
            assert_eq!(new, old, "{}", identified.name());
        }
        struct MissingIdentity;
        impl LogitProcessor for MissingIdentity {
            fn process(&self, _: &mut [f32], _: &ProcessorContext) -> anyhow::Result<()> {
                Ok(())
            }
            fn name(&self) -> &str {
                "missing_identity"
            }
            fn needs_candidate_identity(&self) -> bool {
                true
            }
            fn process_with_candidate_count(
                &self,
                _: &mut [f32],
                _: &ProcessorContext,
                count: usize,
            ) -> anyhow::Result<usize> {
                Ok(count - 1)
            }
        }
        let mut chain = ProcessorChain::new();
        chain.add(Box::new(MissingIdentity));
        let error = format!(
            "{:#}",
            chain
                .process(&mut [0.0, 1.0], &ProcessorContext::default())
                .unwrap_err()
        );
        assert!(
            error.contains("missing_identity") && error.contains("identity contract"),
            "{error}"
        );
    }

    #[test]
    fn infill_prefix_index_matches_rank_ordered_reference_with_byte_fragments() {
        let contract = infill_fixture(
            &[
                b"a",
                b"ab",
                b"abc",
                b"abd",
                b"b",
                b"a",
                &[0xc3],
                &[0xc3, 0xa9],
                &[0xff],
                &[0xff, b'x'],
                b"",
                b"",
            ],
            vec![10, 11],
            10,
        );
        let mut rng = StdRng::seed_from_u64(191);
        for case in 0..160 {
            let mut original: Vec<f32> = (0..12).map(|_| rng.random_range(-4..=4) as f32).collect();
            original[10] = if case % 4 == 0 { 1.0 } else { -20.0 };
            original[11] = -25.0;
            if case % 3 == 0 {
                original[3] = f32::NEG_INFINITY;
                original[8] = f32::NEG_INFINITY;
            }
            let mut expected = original.clone();
            let mut actual = original.clone();
            let mut expected_set = SamplingCandidates::new(&original);
            let mut actual_set = SamplingCandidates::new(&original);
            reference_infill(&contract, &mut expected, &mut expected_set).unwrap();
            InfillProcessor {
                contract: contract.clone(),
            }
            .process_with_candidates(&mut actual, &ProcessorContext::default(), &mut actual_set)
            .unwrap();
            expected_set.indices.sort_unstable();
            actual_set.indices.sort_unstable();
            assert_eq!(actual_set.indices, expected_set.indices, "case {case}");
            assert_eq!(actual, expected, "case {case}");
        }
    }

    #[test]
    fn infill_fallback_prefers_declared_eot_and_preserves_mandatory_masks() {
        let contract = infill_fixture(
            &[b"a", b"b", b"c", b"d", b"e", b"f", b"", b""],
            vec![6, 7],
            7,
        );
        let original = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -20.0, -20.0];
        let mut candidates = SamplingCandidates::new(&original);
        let mut logits = original.clone();
        TopKProcessor { top_k: 6 }
            .process_with_candidates(&mut logits, &ProcessorContext::default(), &mut candidates)
            .unwrap();
        assert!(!candidates.indices.contains(&7));
        InfillProcessor {
            contract: contract.clone(),
        }
        .process_with_candidates(&mut logits, &ProcessorContext::default(), &mut candidates)
        .unwrap();
        assert_eq!(candidates.indices, [7]);
        assert_eq!(logits[7], 1.0);
        assert!(logits[..7].iter().all(|v| *v == f32::NEG_INFINITY));
        let mut logits = original;
        logits[7] = f32::NEG_INFINITY;
        let mut candidates = SamplingCandidates::new(&logits);
        TopKProcessor { top_k: 6 }
            .process_with_candidates(&mut logits, &ProcessorContext::default(), &mut candidates)
            .unwrap();
        let error = InfillProcessor { contract }
            .process_with_candidates(&mut logits, &ProcessorContext::default(), &mut candidates)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("mandatory") && error.contains("7"),
            "{error}"
        );
    }

    #[test]
    fn infill_identity_tracks_masked_eos_through_dynamic_temperature() {
        let contract = infill_fixture(&[b"a", b"b", b"", b"d"], vec![2], 2);
        let make = |top_k| {
            let mut chain = ProcessorChain::new();
            chain.begin_optional_stages();
            if top_k {
                chain.add(Box::new(TopKProcessor { top_k: 2 }));
            } else {
                chain.add(Box::new(TopNSigmaProcessor { n: 0.01 }));
            }
            chain.add(Box::new(InfillProcessor {
                contract: contract.clone(),
            }));
            chain.add(Box::new(TemperatureProcessor {
                temperature: 1.0,
                dynamic: Some(crate::DynamicTemperatureConfig {
                    range: 0.5,
                    exponent: 1.0,
                }),
            }));
            chain
        };
        let original = vec![3.0, 3.0, 0.0, -100.0];
        let mut truncated = original.clone();
        make(true)
            .process(&mut truncated, &ProcessorContext::default())
            .unwrap();
        assert_eq!(truncated[..2], [2.0, 2.0]);
        let mut masked = original;
        make(false)
            .process(&mut masked, &ProcessorContext::default())
            .unwrap();
        let expected = 3.0 / (0.5 + 2.0_f32.ln() / 3.0_f32.ln());
        assert!((masked[0] - expected).abs() < 1e-5, "{masked:?}");
    }

    #[test]
    fn infill_merges_large_duplicate_byte_pieces_with_stable_id_ties() {
        let mut pieces = vec![b"a".to_vec(); 4096];
        pieces.push(Vec::new());
        let contract = std::sync::Arc::new(
            InfillContract::from_pieces(
                (0..pieces.len() as u32).collect(),
                pieces,
                vec![4096],
                4096,
            )
            .unwrap(),
        );
        let mut logits = vec![0.0; 4097];
        logits[4096] = -100.0;
        let mut candidates = SamplingCandidates::new(&logits);
        InfillProcessor { contract }
            .process_with_candidates(&mut logits, &ProcessorContext::default(), &mut candidates)
            .unwrap();
        assert_eq!(candidates.indices, [0, 4096]);
        assert_eq!(logits[0], 0.0);
        assert_eq!(logits[4096], -100.0);
    }

    #[test]
    fn adaptive_p_feedback_uses_original_selected_probability_and_restores_branches() {
        let processor = AdaptivePProcessor::new(crate::AdaptivePConfig {
            target: 0.6,
            decay: 0.0,
        })
        .unwrap();
        let original = vec![0.8_f32.ln(), 0.2_f32.ln(), f32::NEG_INFINITY];
        let mut context = ProcessorContext::default();
        let mut first = original.clone();
        processor.process(&mut first, &context).unwrap();
        assert!((first[0] - 2.3333333).abs() < 1e-5);
        assert!((first[1] - -2.6190476).abs() < 1e-5);
        assert!(first[0] > first[1]);
        assert_eq!(first[2], f32::NEG_INFINITY);
        processor.on_selected(0, &context).unwrap();
        context.generated_tokens.push(0);
        let mut second = original.clone();
        processor.process(&mut second, &context).unwrap();
        assert!(
            second[1] > second[0],
            "accepted original probability 0.8 adapts target to 0.4"
        );
        assert!((processor.state.lock().unwrap().stats.weighted_sum - 0.8).abs() < 1e-6);
        processor.on_selected(1, &context).unwrap();
        // Discard the staged second token and retry the identical accepted prefix.
        let mut retried = original.clone();
        processor.process(&mut retried, &context).unwrap();
        assert_eq!(retried, second);
        processor.on_selected(0, &context).unwrap();
        context.generated_tokens.push(0);
        let mut next = original.clone();
        processor.process(&mut next, &context).unwrap();
        assert_eq!(
            next, second,
            "discarded probability 0.2 must not leak into EMA"
        );
        // Rewind all choices: the request's initial target is restored.
        context.generated_tokens.clear();
        let mut reset = original;
        processor.process(&mut reset, &context).unwrap();
        assert_eq!(reset, first);
    }

    #[test]
    fn adaptive_p_decay_keeps_the_initialized_weighted_prior() {
        let processor = AdaptivePProcessor::new(crate::AdaptivePConfig {
            target: 0.5,
            decay: 0.9,
        })
        .unwrap();
        let context = ProcessorContext::default();
        let mut logits = [0.8_f32.ln(), 0.2_f32.ln()];
        processor.process(&mut logits, &context).unwrap();
        assert!((logits[0] - logits[1]).abs() < 1e-5);
        processor.on_selected(0, &context).unwrap();
        let state = processor.state.lock().unwrap();
        assert!((state.stats.weighted_sum - 5.3).abs() < 1e-5);
        assert!((state.stats.total_weight - 10.0).abs() < 1e-5);
        drop(state);
        let mut context = context;
        context.generated_tokens.push(0);
        let mut logits = [0.8_f32.ln(), 0.2_f32.ln()];
        processor.process(&mut logits, &context).unwrap();
        assert!((logits[0] - -0.7619048).abs() < 1e-4);
        assert!((logits[1] - 0.7368421).abs() < 1e-4);
    }

    #[test]
    fn adaptive_p_rejects_missing_feedback_and_unusable_distributions() {
        let processor = AdaptivePProcessor::new(crate::AdaptivePConfig {
            target: 0.5,
            decay: 0.9,
        })
        .unwrap();
        let mut logits = vec![0.0, 1.0];
        let error = processor
            .process(&mut logits, &context(vec![], vec![0]))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("without selection feedback") && error.contains("restart"),
            "{error}"
        );
        let empty = ProcessorContext::default();
        processor.process(&mut [0.0, 1.0], &empty).unwrap();
        processor.on_selected(0, &empty).unwrap();
        let accepted = context(vec![], vec![0]);
        processor.process(&mut [0.0, 1.0], &accepted).unwrap();
        let changed = context(vec![], vec![1]);
        let error = processor.on_selected(1, &changed).unwrap_err().to_string();
        assert!(
            error.contains("history changed") && error.contains("restart"),
            "{error}"
        );
        let error = processor
            .process(&mut [f32::NEG_INFINITY], &ProcessorContext::default())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("normalize") && error.contains("finite unmasked"),
            "{error}"
        );
    }

    #[test]
    fn poisoned_sampler_errors_reach_selection_before_sampling() {
        struct NeverSample;
        impl crate::Sampler for NeverSample {
            fn sample(&mut self, _: &[f32], _: &ProcessorContext) -> TokenId {
                panic!("error must stop terminal selection")
            }
            fn name(&self) -> &str {
                "must_not_run"
            }
        }
        let processor = MirostatProcessor::new(5.0, 0.1, MirostatVersion::V2);
        let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = processor.state.lock().unwrap();
            panic!("inject poisoned request state");
        }));
        assert!(poisoned.is_err());
        let mut chain = ProcessorChain::new();
        chain.add(Box::new(processor));
        let error = crate::processors::select_next_token_with_sampler(
            &mut [0.0, 1.0],
            &ProcessorContext::default(),
            &chain,
            &mut NeverSample,
        )
        .unwrap_err();
        let error = format!("{error:#}");
        assert!(
            error.contains("mirostat_v2")
                && error.contains("poisoned")
                && error.contains("restart"),
            "{error}"
        );
    }

    #[test]
    fn repetition_penalty_applies_once_per_seen_token() {
        let processor = PenaltiesProcessor {
            repetition_penalty: 2.0,
            window: None,
            ..Default::default()
        };
        let mut logits = vec![4.0, -4.0, 8.0];
        processor
            .process(&mut logits, &context(vec![0, 0], vec![1, 1]))
            .unwrap();
        assert_eq!(logits, vec![2.0, -8.0, 8.0]);
    }

    #[test]
    fn repetition_penalty_window_limits_to_recent_tokens() {
        // Combined stream is prompt [0,1] ++ generated [2,3]; a window of 2 keeps
        // only the last two tokens (2 and 3), leaving 0 and 1 untouched.
        let processor = PenaltiesProcessor {
            repetition_penalty: 2.0,
            window: Some(2),
            ..Default::default()
        };
        let mut logits = vec![4.0, 4.0, 4.0, -4.0];
        processor
            .process(&mut logits, &context(vec![0, 1], vec![2, 3]))
            .unwrap();
        assert_eq!(logits, vec![4.0, 4.0, 2.0, -8.0]);
    }

    #[test]
    fn repetition_penalty_window_zero_penalizes_nothing() {
        let processor = PenaltiesProcessor {
            repetition_penalty: 2.0,
            window: Some(0),
            ..Default::default()
        };
        let mut logits = vec![4.0, 4.0];
        processor
            .process(&mut logits, &context(vec![0], vec![1]))
            .unwrap();
        assert_eq!(logits, vec![4.0, 4.0]);
    }

    #[test]
    fn repetition_penalty_disabled_is_identity() {
        let processor = PenaltiesProcessor {
            repetition_penalty: 1.0,
            window: None,
            ..Default::default()
        };
        let mut logits = vec![4.0, -4.0, 8.0];
        let original = logits.clone();
        processor
            .process(&mut logits, &context(vec![0, 1], vec![2]))
            .unwrap();
        assert_eq!(logits, original);
    }

    #[test]
    fn frequency_penalty_counts_prompt_and_generated_tokens() {
        let processor = PenaltiesProcessor {
            frequency_penalty: 0.5,
            ..Default::default()
        };
        let mut logits = vec![4.0, 4.0, 4.0];
        processor
            .process(&mut logits, &context(vec![0, 0], vec![0, 1, 1]))
            .unwrap();
        assert_eq!(logits, vec![2.5, 3.0, 4.0]);
    }

    #[test]
    fn presence_penalty_applies_once_per_prompt_or_generated_token() {
        let processor = PenaltiesProcessor {
            presence_penalty: 0.75,
            ..Default::default()
        };
        let mut logits = vec![4.0, 4.0, 4.0];
        processor
            .process(&mut logits, &context(vec![0], vec![1, 1, 2]))
            .unwrap();
        assert_eq!(logits, vec![3.25, 3.25, 3.25]);
    }

    #[test]
    fn penalties_share_one_window_across_prompt_generated_and_all_coefficients() {
        let context = context(vec![0, 1, 0], vec![2, 0]);
        let mut processor = PenaltiesProcessor {
            repetition_penalty: 2.0,
            frequency_penalty: 0.5,
            presence_penalty: 1.0,
            window: Some(3),
        };
        let original = vec![8.0, -4.0, 4.0, 8.0];
        let mut logits = original.clone();
        processor.process(&mut logits, &context).unwrap();
        assert_eq!(logits, vec![2.0, -4.0, 0.5, 8.0]);
        processor.window = None;
        let mut logits = original.clone();
        processor.process(&mut logits, &context).unwrap();
        assert_eq!(logits, vec![1.5, -9.5, 0.5, 8.0]);
        processor.window = Some(0);
        let mut logits = original.clone();
        processor.process(&mut logits, &context).unwrap();
        assert_eq!(logits, original);
        processor.window = Some(3);
        processor.repetition_penalty = 0.5;
        processor.frequency_penalty = -0.5;
        processor.presence_penalty = -1.0;
        let mut logits = original;
        processor.process(&mut logits, &context).unwrap();
        assert_eq!(logits, vec![18.0, -4.0, 9.5, 8.0]);
    }

    #[test]
    fn penalties_subtract_frequency_and_presence_as_one_expression() {
        let processor = PenaltiesProcessor {
            frequency_penalty: 1e20,
            presence_penalty: 1.0,
            ..Default::default()
        };
        let mut logits = vec![1e20];
        processor
            .process(&mut logits, &context(vec![0], vec![]))
            .unwrap();
        assert_eq!(logits, [0.0]);
        assert_ne!((1e20_f32 - 1e20_f32) - 1.0, logits[0]);
    }

    #[test]
    fn top_k_masks_tokens_below_threshold() {
        let processor = TopKProcessor { top_k: 2 };
        let mut logits = vec![0.0, 5.0, 1.0, 4.0];
        processor
            .process(&mut logits, &ProcessorContext::default())
            .unwrap();
        assert_eq!(logits, vec![f32::NEG_INFINITY, 5.0, f32::NEG_INFINITY, 4.0]);
    }

    #[test]
    fn top_k_keeps_exactly_k_ties_in_both_selection_algorithms() {
        for len in [8, 1024] {
            for k in [1, 3, 256, 257, 512] {
                if k >= len {
                    continue;
                }
                let mut logits = vec![2.0; len];
                logits[0] = f32::NAN;
                logits[len - 1] = 3.0;
                let mut expected = logits.clone();
                full_sort_top_k(&mut expected, k);
                TopKProcessor { top_k: k }
                    .process(&mut logits, &ProcessorContext::default())
                    .unwrap();
                assert_eq!(logits, expected, "len={len} k={k}");
                assert_eq!(logits.iter().filter(|value| value.is_finite()).count(), k);
                assert!(logits[len - 1].is_finite());
                for p in [0.5, 1.0] {
                    let mut sequential = expected.clone();
                    TopPProcessor { top_p: p }
                        .process(&mut sequential, &ProcessorContext::default())
                        .unwrap();
                    let mut fused = vec![2.0; len];
                    fused[0] = f32::NAN;
                    fused[len - 1] = 3.0;
                    TopKTopPProcessor { top_k: k, top_p: p }
                        .process(&mut fused, &ProcessorContext::default())
                        .unwrap();
                    assert_eq!(fused, sequential, "len={len} k={k} p={p}");
                }
            }
        }
        for row in [
            vec![-0.0, 0.0, -0.0],
            vec![f32::INFINITY, f32::INFINITY, 1.0],
        ] {
            let mut actual = row.clone();
            TopKProcessor { top_k: 1 }
                .process(&mut actual, &ProcessorContext::default())
                .unwrap();
            assert_eq!(actual[0].to_bits(), row[0].to_bits());
            assert_eq!(actual[1..], [f32::NEG_INFINITY, f32::NEG_INFINITY]);
        }
    }

    #[test]
    fn top_k_selection_threshold_matches_full_sort_on_large_vocab() {
        let logits = deterministic_logits(10_000);
        for top_k in [1, 2, 20, 128, 1_024, 9_999] {
            let expected = full_sort_top_k_threshold(&logits, top_k);
            let actual = top_k_boundary(&logits, top_k).map(|boundary| boundary.threshold);
            assert_eq!(actual, expected, "top_k={top_k}");

            let mut expected_logits = logits.clone();
            let mut actual_logits = logits.clone();
            full_sort_top_k(&mut expected_logits, top_k);
            TopKProcessor { top_k }
                .process(&mut actual_logits, &ProcessorContext::default())
                .unwrap();
            assert_eq!(actual_logits, expected_logits, "top_k={top_k}");
        }
    }

    #[test]
    fn top_p_keeps_minimal_nucleus_and_at_least_one_token() {
        let processor = TopPProcessor { top_p: 0.6 };
        let mut logits = vec![3.0, 2.0, 1.0];
        processor
            .process(&mut logits, &ProcessorContext::default())
            .unwrap();
        assert!(logits[0].is_finite());
        assert_eq!(logits[1], f32::NEG_INFINITY);
        assert_eq!(logits[2], f32::NEG_INFINITY);
    }

    #[test]
    fn top_p_fast_path_keeps_same_nucleus_as_full_sort() {
        let logits = deterministic_logits(10_000);
        for top_p in [0.0, 0.25, 0.6, 0.95, 0.999] {
            let mut expected = logits.clone();
            let mut actual = logits.clone();
            full_sort_top_p(&mut expected, top_p);
            TopPProcessor { top_p }
                .process(&mut actual, &ProcessorContext::default())
                .unwrap();
            assert_eq!(actual, expected, "top_p={top_p}");
        }
    }

    /// The fused processor must be indistinguishable from the two it replaces.
    ///
    /// This is the whole safety argument for the fusion: it is a performance
    /// change only if the surviving token set is identical for every input,
    /// including the awkward ones (ties at the top-k threshold, NaN, `-inf`,
    /// k larger than the vocabulary, cutoffs at the extremes).
    #[test]
    fn the_fused_top_k_top_p_matches_running_the_two_processors_in_sequence() {
        let mut state = 0x5EED_1234_ABCD_0001_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };

        for case in 0..200 {
            let len = match case % 4 {
                0 => 16 + (next() * 40.0) as usize,
                1 => 128,
                _ => 400 + (next() * 2000.0) as usize,
            };
            let mut logits: Vec<f32> = (0..len).map(|_| (next() as f32) * 30.0 - 15.0).collect();

            // Deliberately introduce ties, NaN and -inf: the fused path handles
            // exclusion itself instead of inheriting it from TopKProcessor, so
            // these are exactly where the two could drift apart.
            if case % 3 == 0 && len > 8 {
                let tied = logits[1];
                logits[2] = tied;
                logits[3] = tied;
            }
            if case % 5 == 0 && len > 6 {
                logits[4] = f32::NAN;
                logits[5] = f32::NEG_INFINITY;
            }

            for &top_k in &[1usize, 8, 40, 512] {
                for &top_p in &[0.1f32, 0.9, 0.999] {
                    let mut sequential = logits.clone();
                    TopKProcessor { top_k }
                        .process(&mut sequential, &ProcessorContext::default())
                        .unwrap();
                    TopPProcessor { top_p }
                        .process(&mut sequential, &ProcessorContext::default())
                        .unwrap();

                    let mut fused = logits.clone();
                    TopKTopPProcessor { top_k, top_p }
                        .process(&mut fused, &ProcessorContext::default())
                        .unwrap();

                    // Compare surviving sets rather than raw bits: both paths
                    // leave excluded entries as -inf and survivors untouched,
                    // so the surviving index set is the observable behaviour.
                    let sequential_kept: Vec<usize> = sequential
                        .iter()
                        .enumerate()
                        .filter(|(_, value)| value.is_finite())
                        .map(|(index, _)| index)
                        .collect();
                    let fused_kept: Vec<usize> = fused
                        .iter()
                        .enumerate()
                        .filter(|(_, value)| value.is_finite())
                        .map(|(index, _)| index)
                        .collect();

                    assert_eq!(
                        fused_kept, sequential_kept,
                        "fused diverged: case={case} len={len} top_k={top_k} top_p={top_p}"
                    );
                }
            }
        }
    }

    /// The softmax helper must not allocate more than the result it returns.
    ///
    /// This runs once per generated token, so a staging vector here is paid on
    /// every step of every request. An earlier version collected `(index,
    /// logit)` pairs purely to find the maximum and then consumed them, which
    /// on a 151k vocabulary is an extra ~1.2 MB filled and discarded per token.
    ///
    /// Capacity is the observable proxy for that: `weights` must be reserved
    /// once for the whole vocabulary rather than grown, and the zero list must
    /// not allocate when there is nothing to put in it.
    #[test]
    fn the_softmax_helper_reserves_once_and_does_not_allocate_an_empty_zero_list() {
        let logits: Vec<f32> = (0..4096).map(|i| (i % 97) as f32 * 0.1 - 4.0).collect();
        let (weights, zeros) =
            nonzero_softmax_probabilities(&logits).expect("finite logits produce a distribution");

        assert_eq!(
            weights.capacity(),
            logits.len(),
            "weights must be reserved once for the whole vocabulary, not grown"
        );
        assert!(
            zeros.is_empty() && zeros.capacity() == 0,
            "no NaN and no -inf means the zero list must never allocate, got \
             len={} capacity={}",
            zeros.len(),
            zeros.capacity()
        );
        assert_eq!(weights.len(), logits.len(), "every finite logit survives");

        let total: f32 = weights.iter().map(|&(_, p)| p).sum();
        assert!(
            (total - 1.0).abs() < 1e-4,
            "probabilities must be normalised, got {total}"
        );
    }

    /// Excluded logits still reach the caller so it can zero them.
    #[test]
    fn the_softmax_helper_reports_nan_and_neg_infinity_correctly() {
        let logits = vec![1.0, f32::NAN, 2.0, f32::NEG_INFINITY, 0.5];
        let (weights, zeros) = nonzero_softmax_probabilities(&logits).expect("some logits finite");

        // NaN is reported for zeroing; -inf already has zero probability and is
        // simply absent from the weights.
        assert_eq!(zeros, vec![1], "NaN indices must be reported");
        let kept: Vec<usize> = weights.iter().map(|&(index, _)| index).collect();
        assert_eq!(kept, vec![0, 2, 4], "-inf must be dropped, order preserved");

        let total: f32 = weights.iter().map(|&(_, p)| p).sum();
        assert!((total - 1.0).abs() < 1e-5, "got {total}");
    }
    /// across many distributions, not just the one this file happens to build.
    ///
    /// Both paths sum the same probabilities in different orders, and
    /// floating-point addition is not associative, so the boundary is where
    /// they can disagree. A single fixed distribution does not exercise that:
    /// an earlier version of this test passed on `deterministic_logits` while
    /// the two paths differed by one token on roughly 0.2% of random inputs,
    /// which changed the sampled token for a narrow band of RNG values.
    ///
    /// The oracle is written out locally in `f64` rather than calling
    /// `sorted_top_p_keep_count`, deliberately. Sharing that helper would make
    /// the reference degrade in lockstep with the implementation: narrowing
    /// both accumulators back to `f32` would then still pass, which is exactly
    /// the regression this test exists to catch. Verified by mutation.
    ///
    /// Sweeps the cutoffs where the disagreement concentrated (0.95-0.999) and
    /// straddles the 64-element threshold between the sorted and partitioned
    /// paths.
    #[test]
    fn top_p_nucleus_matches_a_strict_descending_sum_across_random_distributions() {
        /// Independent oracle: sort descending, accumulate in f64, stop at the
        /// first index whose running total reaches the cutoff.
        fn descending_nucleus(probabilities: &[(usize, f32)], cutoff: f64) -> usize {
            let mut sorted = probabilities.to_vec();
            sorted.sort_unstable_by(compare_ranked_probability);
            let mut cumulative = 0.0_f64;
            for (offset, &(_, probability)) in sorted.iter().enumerate() {
                cumulative += f64::from(probability);
                if cumulative >= cutoff {
                    return offset + 1;
                }
            }
            sorted.len()
        }

        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };

        // 400 cases weighted towards the sizes where the recursion runs.
        // Mutation-verified: narrowing the accumulators back to f32 makes this
        // fail by case 44, so this is comfortably enough to catch a regression
        // while staying fast. An earlier draft used a quarter as many cases
        // spread over more size buckets and passed even when mutated, which
        // would have shipped a test that could not fail.
        for case in 0..400 {
            let len = match case % 5 {
                0 => 8 + (next() * 50.0) as usize,
                1 => 64,
                _ => 500 + (next() * 3500.0) as usize,
            };
            let raw: Vec<f32> = (0..len).map(|_| (next() as f32) * 40.0 - 20.0).collect();
            let max = raw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = raw.iter().map(|&l| (l - max).exp()).collect();
            let total: f32 = exps.iter().sum();
            let probabilities: Vec<(usize, f32)> = exps
                .iter()
                .enumerate()
                .map(|(index, &e)| (index, e / total))
                .collect();

            for cutoff in [0.1_f32, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999] {
                let expected = descending_nucleus(&probabilities, f64::from(cutoff));

                let mut candidate = probabilities.clone();
                let actual = top_p_keep_count(&mut candidate, cutoff);

                assert_eq!(
                    actual, expected,
                    "nucleus boundary diverged: len={len} cutoff={cutoff} \
                     (case {case}); the partitioned sum must agree exactly with \
                     a strict descending sum"
                );
            }
        }
    }

    /// The kept nucleus must actually reach the cutoff, and be minimal.
    ///
    /// Equivalence to the reference is not enough on its own: if both paths
    /// were wrong in the same direction this would still pass. This pins the
    /// property the nucleus is defined by.
    #[test]
    fn the_kept_nucleus_is_the_smallest_prefix_reaching_the_cutoff() {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };

        for _ in 0..200 {
            let len = 65 + (next() * 1000.0) as usize;
            let raw: Vec<f32> = (0..len).map(|_| (next() as f32) * 30.0 - 15.0).collect();
            let max = raw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = raw.iter().map(|&l| (l - max).exp()).collect();
            let total: f32 = exps.iter().sum();
            let probabilities: Vec<(usize, f32)> = exps
                .iter()
                .enumerate()
                .map(|(index, &e)| (index, e / total))
                .collect();

            for cutoff in [0.5_f32, 0.9, 0.99] {
                let mut candidate = probabilities.clone();
                let keep = top_p_keep_count(&mut candidate, cutoff);

                let mut descending = probabilities.clone();
                descending.sort_unstable_by(compare_ranked_probability);
                let mass_at: Vec<f64> = descending
                    .iter()
                    .scan(0.0_f64, |acc, &(_, p)| {
                        *acc += f64::from(p);
                        Some(*acc)
                    })
                    .collect();

                assert!(
                    mass_at[keep - 1] >= f64::from(cutoff),
                    "kept {keep} of {len} but mass {} < cutoff {cutoff}",
                    mass_at[keep - 1]
                );
                if keep > 1 {
                    assert!(
                        mass_at[keep - 2] < f64::from(cutoff),
                        "kept {keep} of {len} but {} tokens already reached the cutoff",
                        keep - 1
                    );
                }
            }
        }
    }

    #[test]
    fn fixed_seed_sampling_matches_full_sort_top_k_top_p_chain() {
        let logits = deterministic_logits(10_000);
        let mut expected = logits.clone();
        let mut actual = logits.clone();

        full_sort_top_k(&mut expected, 20);
        full_sort_top_p(&mut expected, 0.95);
        TopKProcessor { top_k: 20 }
            .process(&mut actual, &ProcessorContext::default())
            .unwrap();
        TopPProcessor { top_p: 0.95 }
            .process(&mut actual, &ProcessorContext::default())
            .unwrap();

        assert_eq!(actual, expected);
        assert_eq!(
            sample_categorical(&actual, 0.424_242),
            sample_categorical(&expected, 0.424_242)
        );
    }

    #[test]
    fn min_p_masks_relative_to_top_token_probability() {
        let processor = MinPProcessor { min_p: 0.5 };
        let mut logits = vec![0.0, -0.5, -1.0];
        processor
            .process(&mut logits, &ProcessorContext::default())
            .unwrap();
        assert!(logits[0].is_finite());
        assert!(logits[1].is_finite());
        assert_eq!(logits[2], f32::NEG_INFINITY);
    }

    #[test]
    fn top_a_uses_squared_top_probability_threshold_inclusively() {
        let processor = TopAProcessor { top_a: 0.5 };
        let mut logits = vec![0.6_f32.ln(), 0.3_f32.ln(), 0.1_f32.ln()];
        processor
            .process(&mut logits, &ProcessorContext::default())
            .unwrap();
        assert!(logits[0].is_finite());
        assert!(logits[1].is_finite());
        assert_eq!(logits[2], f32::NEG_INFINITY);

        let mut boundary = vec![0.5_f32.ln(), 0.25_f32.ln(), 0.25_f32.ln()];
        TopAProcessor { top_a: 1.0 }
            .process(&mut boundary, &ProcessorContext::default())
            .unwrap();
        assert!(boundary.iter().all(|logit| logit.is_finite()));
    }

    #[test]
    fn top_a_disabled_equal_and_single_token_distributions_are_safe() {
        let original = vec![0.0, 0.0, 0.0];
        let mut disabled = original.clone();
        TopAProcessor { top_a: 0.0 }
            .process(&mut disabled, &ProcessorContext::default())
            .unwrap();
        assert_eq!(disabled, original);

        let mut equal = vec![0.0, 0.0, 0.0, 0.0];
        TopAProcessor { top_a: 1.0 }
            .process(&mut equal, &ProcessorContext::default())
            .unwrap();
        assert!(equal.iter().all(|logit| logit.is_finite()));

        let mut single = vec![4.0];
        TopAProcessor { top_a: 1.0 }
            .process(&mut single, &ProcessorContext::default())
            .unwrap();
        assert_eq!(single, vec![4.0]);
    }

    #[test]
    fn typical_p_keeps_tokens_nearest_entropy_until_mass_is_reached() {
        // For [0.5, 0.3, 0.2], token 1 is most typical, followed by 0, then 2.
        let processor = TypicalPProcessor { typical_p: 0.6 };
        let mut logits = vec![0.5_f32.ln(), 0.3_f32.ln(), 0.2_f32.ln()];
        processor
            .process(&mut logits, &ProcessorContext::default())
            .unwrap();
        assert!(logits[0].is_finite());
        assert!(logits[1].is_finite());
        assert_eq!(logits[2], f32::NEG_INFINITY);
    }

    #[test]
    fn typical_p_handles_noop_equal_and_single_token_distributions() {
        let original = vec![1.0, 0.0];
        let mut disabled = original.clone();
        TypicalPProcessor { typical_p: 1.0 }
            .process(&mut disabled, &ProcessorContext::default())
            .unwrap();
        assert_eq!(disabled, original);

        let mut equal = vec![0.0; 4];
        TypicalPProcessor { typical_p: 0.5 }
            .process(&mut equal, &ProcessorContext::default())
            .unwrap();
        assert!(equal[0].is_finite());
        assert!(equal[1].is_finite());
        assert_eq!(equal[2], f32::NEG_INFINITY);
        assert_eq!(equal[3], f32::NEG_INFINITY);

        let mut single = vec![0.0];
        TypicalPProcessor { typical_p: 0.1 }
            .process(&mut single, &ProcessorContext::default())
            .unwrap();
        assert_eq!(single, vec![0.0]);
    }

    #[test]
    fn mirostat_v1_estimates_zipf_slope_and_keeps_expected_top_k() {
        // p(rank) proportional to rank^-2 gives slope=2. With mu=2 and n=4,
        // k=round(sqrt(4 / (1 - 4^-1)))=2.
        let processor = MirostatProcessor::new(1.0, 0.1, MirostatVersion::V1);
        let mut logits = vec![0.0, -(4.0_f32).ln(), -(9.0_f32).ln(), -(16.0_f32).ln()];
        processor
            .process(&mut logits, &ProcessorContext::default())
            .unwrap();
        assert!(logits[0].is_finite());
        assert!(logits[1].is_finite());
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[3], f32::NEG_INFINITY);
    }

    #[test]
    fn mirostat_v2_state_persists_and_surprise_converges_toward_tau() {
        let processor = MirostatProcessor::new(2.0, 0.5, MirostatVersion::V2);
        let probabilities = [0.5_f32, 0.25, 0.125, 0.0625, 0.0625];
        let base_logits: Vec<_> = probabilities
            .iter()
            .map(|probability| probability.ln())
            .collect();
        let mut generated_tokens = Vec::new();
        let mut observed_surprises = Vec::new();
        let mut mu_by_step = Vec::new();

        for _ in 0..5 {
            let mut logits = base_logits.clone();
            processor
                .process(
                    &mut logits,
                    &ProcessorContext {
                        generated_tokens: generated_tokens.clone(),
                        ..Default::default()
                    },
                )
                .unwrap();
            mu_by_step.push(processor.current_mu());

            let selected = logits
                .iter()
                .rposition(|logit| logit.is_finite())
                .expect("Mirostat must retain a sampleable token");
            observed_surprises.push(-probabilities[selected].log2());
            generated_tokens.push(selected as TokenId);
        }

        assert_eq!(observed_surprises, vec![4.0, 3.0, 2.0, 2.0, 2.0]);
        assert!((mu_by_step[0] - 4.0).abs() < f32::EPSILON);
        assert!((mu_by_step[1] - 3.0).abs() < f32::EPSILON);
        assert!((mu_by_step[2] - 2.5).abs() < f32::EPSILON);
        assert!(
            mu_by_step.windows(2).all(|pair| pair[1] <= pair[0]),
            "surprise above tau must lower mu across decode steps"
        );
        assert_eq!(&mu_by_step[2..], &[2.5, 2.5, 2.5]);
    }

    #[test]
    fn mirostat_feedback_moves_mu_in_both_directions() {
        let processor = MirostatProcessor::new(0.5, 0.25, MirostatVersion::V2);
        let mut first = vec![0.6_f32.ln(), 0.3_f32.ln(), 0.1_f32.ln()];
        processor
            .process(&mut first, &ProcessorContext::default())
            .unwrap();

        let initial_mu = processor.current_mu();
        let mut second = vec![0.9_f32.ln(), 0.1_f32.ln()];
        processor
            .process(
                &mut second,
                &ProcessorContext {
                    generated_tokens: vec![1],
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(
            processor.current_mu() < initial_mu,
            "a token more surprising than tau must lower mu"
        );

        let lowered_mu = processor.current_mu();
        let mut third = vec![0.9_f32.ln(), 0.1_f32.ln()];
        processor
            .process(
                &mut third,
                &ProcessorContext {
                    generated_tokens: vec![1, 0],
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(
            processor.current_mu() > lowered_mu,
            "a token less surprising than tau must raise mu"
        );
    }

    #[test]
    fn mirostat_single_token_distribution_remains_sampleable() {
        for version in [MirostatVersion::V1, MirostatVersion::V2] {
            let processor = MirostatProcessor::new(5.0, 0.1, version);
            let mut logits = vec![3.0];
            processor
                .process(&mut logits, &ProcessorContext::default())
                .unwrap();
            assert_eq!(logits, vec![3.0]);
        }
    }

    #[test]
    fn dry_penalizes_only_continuations_of_repeated_suffixes() {
        let processor = DryProcessor {
            multiplier: 2.0,
            base: 1.5,
            allowed_length: 2,
            sequence_breakers: vec![],
            window: None,
        };
        let mut logits = vec![0.0; 5];
        processor
            .process(&mut logits, &context(vec![1, 2, 3], vec![1, 2]))
            .unwrap();
        assert_eq!(logits[3], -2.0);
        assert_eq!(logits[0], 0.0);
        assert_eq!(logits[1], 0.0);
        assert_eq!(logits[2], 0.0);
        assert_eq!(logits[4], 0.0);
    }

    #[test]
    fn dry_empty_and_short_contexts_are_noops() {
        let processor = DryProcessor {
            multiplier: 2.0,
            base: 1.5,
            allowed_length: 2,
            sequence_breakers: vec![],
            window: None,
        };
        let original = vec![1.0; 5];

        let mut empty = original.clone();
        processor
            .process(&mut empty, &ProcessorContext::default())
            .unwrap();
        assert_eq!(empty, original);

        let mut short = original.clone();
        processor
            .process(&mut short, &context(vec![1], vec![2]))
            .unwrap();
        assert_eq!(short, original);
    }

    #[test]
    fn dry_penalty_grows_for_longer_repeated_suffix() {
        let mut logits = vec![0.0; 10];
        DryProcessor {
            multiplier: 2.0,
            base: 1.5,
            allowed_length: 2,
            sequence_breakers: vec![],
            window: None,
        }
        .process(&mut logits, &context(vec![1, 2, 3, 4], vec![1, 2, 3]))
        .unwrap();
        assert_eq!(logits[4], -3.0);
    }

    #[test]
    fn dry_sequence_breaker_resets_suffix_matching() {
        let prompt_tokens = vec![1, 9, 2, 3];
        let generated_tokens = vec![1, 9, 2];
        let mut unbroken = vec![0.0; 10];
        DryProcessor {
            multiplier: 2.0,
            base: 1.5,
            allowed_length: 2,
            sequence_breakers: vec![],
            window: None,
        }
        .process(
            &mut unbroken,
            &context(prompt_tokens.clone(), generated_tokens.clone()),
        )
        .unwrap();
        assert_eq!(unbroken[3], -3.0);

        let mut broken = vec![0.0; 10];
        DryProcessor {
            multiplier: 2.0,
            base: 1.5,
            allowed_length: 2,
            sequence_breakers: vec![vec![9]],
            window: None,
        }
        .process(&mut broken, &context(prompt_tokens, generated_tokens))
        .unwrap();
        assert!(broken.iter().all(|logit| *logit == 0.0));
    }

    #[test]
    fn dry_multi_token_breakers_preserve_partial_matches_and_honor_history_window() {
        let mut processor = DryProcessor {
            multiplier: 2.0,
            base: 1.5,
            allowed_length: 2,
            sequence_breakers: vec![vec![9, 8]],
            window: None,
        };
        let history = vec![1, 9, 2, 3, 1, 9, 2];
        let mut logits = vec![0.0; 10];
        processor
            .process(&mut logits, &context(history.clone(), vec![]))
            .unwrap();
        assert_eq!(logits[3], -3.0);
        processor.window = Some(4);
        let mut logits = vec![0.0; 10];
        processor
            .process(&mut logits, &context(history, vec![]))
            .unwrap();
        assert!(logits.iter().all(|value| *value == 0.0));
        processor.window = None;
        processor
            .process(
                &mut logits,
                &context(vec![1, 9, 8, 2, 3, 1, 9, 8, 2], vec![]),
            )
            .unwrap();
        assert!(logits.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn dry_zero_multiplier_is_noop() {
        let mut logits = vec![1.0; 4];
        let original = logits.clone();
        DryProcessor {
            multiplier: 0.0,
            base: 1.75,
            allowed_length: 2,
            sequence_breakers: vec![],
            window: None,
        }
        .process(&mut logits, &context(vec![1, 2, 3], vec![1, 2]))
        .unwrap();
        assert_eq!(logits, original);
    }

    #[test]
    fn xtc_probability_one_keeps_exact_expected_set() {
        let processor = XtcProcessor::new(1.0, 0.25, Some(7));
        let mut logits = vec![0.5_f32.ln(), 0.3_f32.ln(), 0.2_f32.ln()];
        processor
            .process(&mut logits, &ProcessorContext::default())
            .unwrap();
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert!(logits[1].is_finite());
        assert!(logits[2].is_finite());
        assert_eq!(
            logits
                .iter()
                .enumerate()
                .filter_map(|(index, logit)| logit.is_finite().then_some(index))
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    #[test]
    fn xtc_threshold_zero_never_admits_masked_tokens_or_masks_every_valid_token() {
        let processor = XtcProcessor::new(1.0, 0.0, Some(7));
        let mut logits = vec![0.7_f32.ln(), 0.3_f32.ln(), f32::NEG_INFINITY];
        processor
            .process(&mut logits, &ProcessorContext::default())
            .unwrap();

        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert!(logits[1].is_finite());
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits.iter().filter(|logit| logit.is_finite()).count(), 1);
    }

    #[test]
    fn xtc_excludes_probability_exactly_at_threshold() {
        let processor_logits = vec![0.4_f32.ln(), 0.3_f32.ln(), 0.2_f32.ln(), 0.1_f32.ln()];
        let threshold = softmax_probabilities(&processor_logits).expect("valid distribution")[2];
        let processor = XtcProcessor::new(1.0, threshold, Some(7));
        let mut logits = processor_logits;
        processor
            .process(&mut logits, &ProcessorContext::default())
            .unwrap();

        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert!(logits[1].is_finite());
        assert!(logits[2].is_finite(), "the boundary token is excluded");
        assert!(logits[3].is_finite());
    }

    #[test]
    fn xtc_probability_zero_is_noop() {
        let original = vec![2.0, 1.0];
        let mut disabled = original.clone();
        XtcProcessor::new(0.0, 0.1, Some(7))
            .process(&mut disabled, &ProcessorContext::default())
            .unwrap();
        assert_eq!(disabled, original);
    }

    #[test]
    fn xtc_one_token_above_threshold_is_noop() {
        let original = vec![0.8_f32.ln(), 0.15_f32.ln(), 0.05_f32.ln()];
        let mut logits = original.clone();
        XtcProcessor::new(1.0, 0.5, Some(7))
            .process(&mut logits, &ProcessorContext::default())
            .unwrap();
        assert_eq!(logits, original);
    }

    #[test]
    fn xtc_uses_seeded_reproducible_probability_draws() {
        let left = XtcProcessor::new(0.5, 0.2, Some(99));
        let right = XtcProcessor::new(0.5, 0.2, Some(99));
        let run = |processor: &XtcProcessor| {
            (0..32)
                .map(|_| {
                    let mut logits = vec![0.5_f32.ln(), 0.3_f32.ln(), 0.2_f32.ln()];
                    processor
                        .process(&mut logits, &ProcessorContext::default())
                        .unwrap();
                    logits
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(run(&left), run(&right));
    }

    #[test]
    fn dynamic_temperature_resolves_entropy_interval_and_exponent() {
        let processor = TemperatureProcessor {
            temperature: 1.0,
            dynamic: Some(crate::config::DynamicTemperatureConfig {
                range: 0.5,
                exponent: 2.0,
            }),
        };
        let logits = [0.75_f32.ln(), 0.25_f32.ln()];
        let entropy = -0.75_f32 * 0.75_f32.ln() - 0.25_f32 * 0.25_f32.ln();
        let expected = 0.5 + (entropy / 2_f32.ln()).powi(2);
        assert!((processor.resolved_temperature(&logits, logits.len()) - expected).abs() < 1e-6);
        let mut scaled = logits;
        processor
            .process(&mut scaled, &ProcessorContext::default())
            .unwrap();
        for (actual, original) in scaled.into_iter().zip(logits) {
            assert!((actual - original / expected).abs() < 1e-6);
        }
        assert!((processor.resolved_temperature(&[3.0, 3.0], 2) - 1.5).abs() < 1e-6);
        assert_eq!(processor.resolved_temperature(&[0.0, -1000.0], 2), 0.5);
    }

    #[test]
    fn dynamic_temperature_handles_zero_entropy_exclusions_and_single_candidate() {
        let mut processor = TemperatureProcessor {
            temperature: 0.5,
            dynamic: Some(crate::config::DynamicTemperatureConfig {
                range: 1.0,
                exponent: 1.0,
            }),
        };
        let mut logits = [0.0, -1000.0, f32::NEG_INFINITY];
        processor
            .process(&mut logits, &ProcessorContext::default())
            .unwrap();
        assert_eq!(logits, [0.0, f32::NEG_INFINITY, f32::NEG_INFINITY]);
        let mut single = [7.0];
        processor
            .process(&mut single, &ProcessorContext::default())
            .unwrap();
        assert_eq!(single, [7.0]);
        processor.dynamic.as_mut().unwrap().exponent = 0.0;
        assert_eq!(processor.resolved_temperature(&[0.0, -1000.0], 2), 1.5);
        processor.dynamic.as_mut().unwrap().range = 0.0;
        assert_eq!(processor.resolved_temperature(&[0.0, -1000.0], 2), 0.5);
    }

    #[test]
    fn temperature_zero_keeps_lowest_id_for_signed_zero_and_infinite_ties() {
        let processor = TemperatureProcessor {
            temperature: 0.0,
            dynamic: None,
        };
        for mut logits in [
            vec![-0.0, 0.0],
            vec![f32::INFINITY, 1.0],
            vec![f32::INFINITY, f32::INFINITY],
        ] {
            let expected = crate::sampling::sample_greedy(&logits);
            processor
                .process(&mut logits, &ProcessorContext::default())
                .unwrap();
            assert_eq!(crate::sampling::sample_greedy(&logits), expected);
        }
    }

    #[test]
    fn temperature_scales_logits() {
        let processor = TemperatureProcessor {
            temperature: 2.0,
            dynamic: None,
        };
        let mut logits = vec![2.0, -4.0];
        processor
            .process(&mut logits, &ProcessorContext::default())
            .unwrap();
        assert_eq!(logits, vec![1.0, -2.0]);
    }

    #[test]
    fn stop_sequence_signals_token_suffix_and_text_suffix() {
        let processor = StopSequenceProcessor::new(vec![
            StopSequence::Tokens(vec![2, 3]),
            StopSequence::Text("END".to_string()),
        ]);
        let token_context = ProcessorContext {
            generated_tokens: vec![1, 2, 3],
            ..Default::default()
        };
        assert_eq!(
            processor.signal(&token_context),
            Some(ProcessorSignal::StopSequence { index: 0 })
        );

        let text_context = ProcessorContext {
            generated_text: "hello END".to_string(),
            ..Default::default()
        };
        assert_eq!(
            processor.signal(&text_context),
            Some(ProcessorSignal::StopSequence { index: 1 })
        );
    }

    fn json_token_texts() -> Vec<Option<String>> {
        vec![
            Some("not-json".to_string()),
            Some("{".to_string()),
            Some("\"".to_string()),
            Some("a".to_string()),
            Some("\":".to_string()),
            Some("1".to_string()),
            Some("}".to_string()),
            Some("[".to_string()),
            Some("true".to_string()),
            Some(",".to_string()),
            Some("null".to_string()),
            Some("]".to_string()),
            Some("\"ok\"".to_string()),
            Some("-12.3e+4".to_string()),
            Some(String::new()),
            Some("\n".to_string()),
        ]
    }

    fn generate_scripted_json(script: &[TokenId], eos_token_id: TokenId) -> String {
        let processor = ConstraintProcessor::new(
            Box::new(JsonConstraint::default()),
            json_token_texts(),
            vec![eos_token_id],
        );
        let mut generated_text = String::new();
        let mut generated_tokens = Vec::new();

        for (step, &desired) in script.iter().enumerate() {
            let context = ProcessorContext {
                generated_tokens: generated_tokens.clone(),
                generated_text: generated_text.clone(),
                step,
                ..Default::default()
            };
            let mut logits = vec![f32::NEG_INFINITY; json_token_texts().len()];
            logits[desired as usize] = 1.0;
            processor.process(&mut logits, &context).unwrap();
            let selected = sample_greedy(&logits);
            assert_eq!(selected, desired);
            if selected == eos_token_id {
                break;
            }
            generated_tokens.push(selected);
            generated_text.push_str(
                json_token_texts()[selected as usize]
                    .as_deref()
                    .expect("test token text"),
            );
        }

        generated_text
    }

    #[test]
    fn prefilled_schema_consumes_exact_bytes_and_keeps_virtual_token_out_of_model_candidates() {
        let bytes = vec![
            b"{".to_vec(),
            b"}".to_vec(),
            b"\"a\":".to_vec(),
            b"1".to_vec(),
            b"true".to_vec(),
            b"<eos>".to_vec(),
        ];
        let schema = r#"{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"],"additionalProperties":false}"#;
        for prefix in ["{", " {\"a\":"] {
            let constraint = LlguidanceConstraint::from_token_bytes_with_prefix(
                GrammarConstraintKind::JsonSchema,
                schema,
                bytes.clone(),
                &[5],
                prefix,
            )
            .unwrap();
            let candidates = bytes
                .iter()
                .enumerate()
                .map(|(id, bytes)| TokenCandidate {
                    token_id: id as u32,
                    text: String::from_utf8(bytes.clone()).unwrap(),
                    is_eos: id == 5,
                })
                .collect::<Vec<_>>();
            let mask = constraint
                .allowed_next_tokens(&ProcessorContext::default(), &candidates)
                .unwrap();
            assert_eq!(mask.len(), bytes.len());
            assert!(
                !mask[1],
                "cannot close an incomplete required property: {mask:?}"
            );
            assert!(!mask[4], "integer schema cannot accept true: {mask:?}");
            assert!(!mask[5]);
            if prefix == "{" {
                assert!(mask[2], "{mask:?}");
            } else {
                assert!(mask[3], "{mask:?}");
            }
        }
        let error = LlguidanceConstraint::from_token_bytes_with_prefix(
            GrammarConstraintKind::JsonSchema,
            schema,
            bytes,
            &[5],
            "{\"b\":",
        )
        .err()
        .expect("undeclared property prefix must fail");
        assert!(error.downcast_ref::<ConstraintPrefixError>().is_some());
        assert!(
            error.chain().count() > 1,
            "preserve the underlying parser cause"
        );
    }

    #[test]
    fn prefilled_schema_chunks_long_utf8_and_does_not_reserve_ordinary_zero_without_eos() {
        let prefix = format!("\"{}", "é".repeat(1024));
        let constraint = LlguidanceConstraint::from_token_bytes_with_prefix(
            GrammarConstraintKind::JsonSchema,
            r#"{"type":"string"}"#,
            vec![b"\"".to_vec(), b"x".to_vec()],
            &[],
            &prefix,
        )
        .unwrap();
        let candidates = vec![
            TokenCandidate {
                token_id: 0,
                text: "\"".into(),
                is_eos: false,
            },
            TokenCandidate {
                token_id: 1,
                text: "x".into(),
                is_eos: false,
            },
        ];
        let mask = constraint
            .allowed_next_tokens(&ProcessorContext::default(), &candidates)
            .unwrap();
        assert_eq!(mask.len(), 2);
        assert!(
            mask[0],
            "ordinary token 0 must be allowed to close the string: {mask:?}"
        );
        assert!(mask[1], "{mask:?}");
    }

    #[test]
    fn prefilled_schema_recognizes_every_declared_ending() {
        let constraint = LlguidanceConstraint::from_token_bytes_with_prefix(
            GrammarConstraintKind::JsonSchema,
            r#"{"type":"integer"}"#,
            vec![b"1".to_vec(), b"<eos>".to_vec(), b"<eot>".to_vec()],
            &[1, 2],
            "1",
        )
        .unwrap();
        let candidates = (0..4)
            .map(|id| TokenCandidate {
                token_id: id,
                text: String::new(),
                is_eos: id == 1 || id == 2,
            })
            .collect::<Vec<_>>();
        let mask = constraint
            .allowed_next_tokens(&ProcessorContext::default(), &candidates)
            .unwrap();
        assert!(
            !mask[3],
            "grammar-only prefix must stay masked even with a padded model-logit candidate: {mask:?}"
        );
        assert!(
            mask[1] && mask[2],
            "all declared endings must be available after a complete value: {mask:?}"
        );
    }

    #[test]
    fn prefilled_schema_rejects_unsupported_vocabulary_without_unwinding_the_worker() {
        let error = LlguidanceConstraint::from_token_bytes_with_prefix(
            GrammarConstraintKind::JsonSchema,
            r#"{"type":"object"}"#,
            vec![vec![b'a'; 2048], b"<eos>".to_vec()],
            &[1],
            "{",
        )
        .err()
        .expect("oversized trie chain must be refused");
        assert!(
            error
                .to_string()
                .contains("repair oversized tokenizer pieces"),
            "{error:#}"
        );
    }

    #[test]
    fn llguidance_history_replay_matches_fresh_parsers_after_rewinds_and_replacements() {
        fn parser() -> LlguidanceConstraint {
            LlguidanceConstraint::from_token_texts(
                GrammarConstraintKind::Regex,
                "a(b|c)",
                &[
                    Some("a".into()),
                    Some("b".into()),
                    Some("c".into()),
                    Some("<eos>".into()),
                ],
                Some(3),
            )
            .unwrap()
        }
        let candidates = (0..4)
            .map(|token_id| TokenCandidate {
                token_id,
                text: String::new(),
                is_eos: token_id == 3,
            })
            .collect::<Vec<_>>();
        let shared = parser();
        for tokens in [
            vec![],
            vec![0],
            vec![0, 1],
            vec![0, 1],
            vec![0, 1, 3],
            vec![0, 1, 3],
            vec![0, 2],
            vec![0],
            vec![],
            vec![0, 1],
        ] {
            let context = ProcessorContext {
                generated_tokens: tokens.clone(),
                ..Default::default()
            };
            let expected = parser().allowed_next_tokens(&context, &candidates).unwrap();
            assert_eq!(
                shared.allowed_next_tokens(&context, &candidates).unwrap(),
                expected,
                "history={tokens:?}"
            );
        }
        let error = shared
            .allowed_next_tokens(
                &ProcessorContext {
                    generated_tokens: vec![0, 0],
                    ..Default::default()
                },
                &candidates,
            )
            .unwrap_err();
        assert!(
            format!("{error:#}").contains("commit history token 0 at offset 1"),
            "{error:#}"
        );
        assert!(error.chain().count() > 1);
        let error = shared
            .allowed_next_tokens(
                &ProcessorContext {
                    generated_tokens: vec![0, 1, 3, 0],
                    ..Default::default()
                },
                &candidates,
            )
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("continues after the grammar completed"),
            "{error:#}"
        );
        let retry = ProcessorContext {
            generated_tokens: vec![0, 2],
            ..Default::default()
        };
        assert_eq!(
            shared.allowed_next_tokens(&retry, &candidates).unwrap(),
            parser().allowed_next_tokens(&retry, &candidates).unwrap()
        );
    }

    #[test]
    fn prefilled_llguidance_replay_preserves_the_original_grammar_prefix() {
        let constraint = LlguidanceConstraint::from_token_bytes_with_prefix(
            GrammarConstraintKind::Regex,
            "a(b|c)",
            vec![
                b"a".to_vec(),
                b"b".to_vec(),
                b"c".to_vec(),
                b"<eos>".to_vec(),
            ],
            &[3],
            "a",
        )
        .unwrap();
        let candidates = (0..4)
            .map(|token_id| TokenCandidate {
                token_id,
                text: String::new(),
                is_eos: token_id == 3,
            })
            .collect::<Vec<_>>();
        for tokens in [vec![], vec![1], vec![], vec![2]] {
            let mask = constraint
                .allowed_next_tokens(
                    &ProcessorContext {
                        generated_tokens: tokens.clone(),
                        ..Default::default()
                    },
                    &candidates,
                )
                .unwrap();
            assert!(!mask[0]);
            assert_eq!(mask[3], !tokens.is_empty());
            if tokens.is_empty() {
                assert!(mask[1] && mask[2]);
            }
        }
    }

    #[test]
    fn llguidance_poisoned_state_reports_an_actionable_error() {
        let constraint = LlguidanceConstraint::from_token_texts(
            GrammarConstraintKind::Regex,
            "a",
            &[Some("a".into())],
            None,
        )
        .unwrap();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = constraint.inner.lock().unwrap();
            panic!("inject grammar state poisoning");
        }));
        assert!(panic.is_err());
        let error = constraint
            .allowed_next_tokens(&ProcessorContext::default(), &[])
            .unwrap_err();
        assert!(
            error.to_string().contains("parser lock is poisoned")
                && error.to_string().contains("fresh grammar state")
        );
    }

    #[test]
    fn lark_suffix_stops_at_first_boundary_even_inside_a_single_token() {
        let grammar = r#"start: reasoning payload
reasoning[suffix="[/r]"]: /(?s:.*)/
payload: %json {"type":"object","properties":{"n":{"const":1}},"required":["n"],"additionalProperties":false}
"#;
        let pieces = [
            "thought",
            "[/r]{\"n\":1}",
            "[/r]bad",
            "text[/r]{\"n\":1}",
            "text[/r]bad[/r]{\"n\":1}",
            "<eos>",
            "text[/",
            "r]{\"n\":1}",
            "r]bad[/r]{\"n\":1}",
        ];
        let constraint = LlguidanceConstraint::from_token_texts(
            GrammarConstraintKind::Lark,
            grammar,
            &pieces
                .iter()
                .map(|piece| Some((*piece).into()))
                .collect::<Vec<_>>(),
            Some(5),
        )
        .unwrap();
        let candidates = pieces
            .iter()
            .enumerate()
            .map(|(id, text)| TokenCandidate {
                token_id: id as u32,
                text: (*text).into(),
                is_eos: id == 5,
            })
            .collect::<Vec<_>>();
        let mask = constraint
            .allowed_next_tokens(&ProcessorContext::default(), &candidates)
            .unwrap();
        assert!(mask[0] && mask[1] && mask[3], "{mask:?}");
        assert!(!mask[2] && !mask[4] && !mask[5], "{mask:?}");
        let mask = constraint
            .allowed_next_tokens(
                &ProcessorContext {
                    generated_tokens: vec![6],
                    ..Default::default()
                },
                &candidates,
            )
            .unwrap();
        assert!(mask[7] && !mask[8], "{mask:?}");
        let mask = constraint
            .allowed_next_tokens(
                &ProcessorContext {
                    generated_tokens: vec![6, 7],
                    ..Default::default()
                },
                &candidates,
            )
            .unwrap();
        assert!(mask[5], "{mask:?}");
    }

    #[test]
    fn lark_suffix_accepts_split_utf8_boundary_bytes_after_reasoning_prefill() {
        let grammar = r#"start: reasoning payload
reasoning[suffix="終"]: /(?s:.*)/
payload: %json {"type":"object","properties":{"n":{"const":1}},"required":["n"],"additionalProperties":false}
"#;
        let constraint = LlguidanceConstraint::from_token_bytes_with_prefix(
            GrammarConstraintKind::Lark,
            grammar,
            vec![
                vec![0xe7, 0xb5],
                b"\x82{\"n\":1}".to_vec(),
                b"\x82bad".to_vec(),
                b"thought".to_vec(),
                b"<eos>".to_vec(),
            ],
            &[4],
            "思考",
        )
        .unwrap();
        let candidates = (0..5)
            .map(|token_id| TokenCandidate {
                token_id,
                text: String::new(),
                is_eos: token_id == 4,
            })
            .collect::<Vec<_>>();
        let mask = constraint
            .allowed_next_tokens(&ProcessorContext::default(), &candidates)
            .unwrap();
        assert!(mask[0] && !mask[4], "{mask:?}");
        let mask = constraint
            .allowed_next_tokens(
                &ProcessorContext {
                    generated_tokens: vec![0],
                    ..Default::default()
                },
                &candidates,
            )
            .unwrap();
        assert!(mask[1] && !mask[2], "{mask:?}");
        let context = ProcessorContext {
            generated_tokens: vec![0, 1],
            ..Default::default()
        };
        let mask = constraint
            .allowed_next_tokens(&context, &candidates)
            .unwrap();
        assert!(mask[4], "{mask:?}");
        assert_eq!(
            constraint
                .allowed_next_tokens(&context, &candidates)
                .unwrap(),
            mask
        );
        let rewind = constraint
            .allowed_next_tokens(&ProcessorContext::default(), &candidates)
            .unwrap();
        assert!(rewind[0] && !rewind[4], "{rewind:?}");
    }

    #[derive(Debug, thiserror::Error)]
    #[error("injected response boundary parsing failure")]
    struct MaskFailure;
    struct FailingMask;
    impl Constraint for FailingMask {
        fn allowed_next_tokens(
            &self,
            _: &ProcessorContext,
            _: &[TokenCandidate],
        ) -> anyhow::Result<Vec<bool>> {
            Err(MaskFailure.into())
        }
        fn name(&self) -> &str {
            "failing_boundary"
        }
    }
    #[test]
    fn constraint_processor_preserves_parser_failures_before_touching_logits() {
        let processor =
            ConstraintProcessor::new(Box::new(FailingMask), vec![Some("a".into())], vec![]);
        let mut logits = vec![2.0];
        let error = processor
            .process(&mut logits, &ProcessorContext::default())
            .unwrap_err();
        assert!(error.is::<MaskFailure>());
        assert!(
            error.to_string().contains("failing_boundary")
                && error.to_string().contains("offset 0")
        );
        assert_eq!(logits, vec![2.0]);
    }

    struct MalformedMask;
    impl Constraint for MalformedMask {
        fn allowed_next_tokens(
            &self,
            _: &ProcessorContext,
            _: &[TokenCandidate],
        ) -> anyhow::Result<Vec<bool>> {
            Ok(vec![])
        }
        fn name(&self) -> &str {
            "malformed_boundary"
        }
    }
    #[test]
    fn constraint_processor_rejects_malformed_mask_dimensions() {
        let processor =
            ConstraintProcessor::new(Box::new(MalformedMask), vec![Some("a".into())], vec![]);
        let mut logits = vec![2.0];
        let error = processor
            .process(&mut logits, &ProcessorContext::default())
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("0 mask entries for 1 candidate logits"),
            "{error:#}"
        );
        assert_eq!(logits, vec![2.0]);
    }

    #[test]
    fn json_continuation_prefix_controls_initial_and_complete_masks() {
        assert!(
            JsonConstraint::new("not JSON".into())
                .unwrap_err()
                .to_string()
                .contains("valid partial JSON")
        );
        let processor = ConstraintProcessor::new(
            Box::new(JsonConstraint::new("{\"a\":".into()).unwrap()),
            json_token_texts(),
            vec![14],
        );
        let mut logits = vec![0.0; json_token_texts().len()];
        processor
            .process(&mut logits, &ProcessorContext::default())
            .unwrap();
        assert!(logits[5].is_finite()); // 1 may complete the pending value.
        assert_eq!(logits[6], f32::NEG_INFINITY); // } cannot omit that value.
        assert_eq!(logits[14], f32::NEG_INFINITY);
        let mut logits = vec![0.0; json_token_texts().len()];
        processor
            .process(
                &mut logits,
                &ProcessorContext {
                    generated_text: "1}".into(),
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(logits[14].is_finite());
        assert_eq!(logits[1], f32::NEG_INFINITY);
    }

    #[test]
    fn grammar_has_no_finite_candidate_returns_an_actionable_error_before_selection() {
        let processor = ConstraintProcessor::new(
            Box::new(JsonConstraint::default()),
            vec![Some("word".into()), Some("<eos>".into())],
            vec![1],
        );
        let error = processor
            .process(&mut [1.0, 2.0], &ProcessorContext::default())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("json_constraint")
                && error.contains("no finite selectable token")
                && error.contains("offset 0")
                && error.contains("tokenizer vocabulary"),
            "{error}"
        );
    }

    #[test]
    fn json_constraint_masks_invalid_tokens_and_allows_eos_only_when_complete() {
        let processor = ConstraintProcessor::new(
            Box::new(JsonConstraint::default()),
            json_token_texts(),
            vec![14],
        );

        let mut logits = vec![0.0; json_token_texts().len()];
        logits[6] = 10.0;
        logits[1] = 1.0;
        processor
            .process(&mut logits, &ProcessorContext::default())
            .unwrap();
        assert_eq!(logits[6], f32::NEG_INFINITY);
        assert!(logits[1].is_finite());
        assert_eq!(logits[14], f32::NEG_INFINITY);

        let complete_context = ProcessorContext {
            generated_text: "{\"a\":1}".to_string(),
            ..Default::default()
        };
        let mut complete_logits = vec![0.0; json_token_texts().len()];
        processor
            .process(&mut complete_logits, &complete_context)
            .unwrap();
        assert!(complete_logits[14].is_finite());
        assert_eq!(complete_logits[1], f32::NEG_INFINITY);
    }

    #[test]
    fn json_constraint_generates_parseable_balanced_json_values() {
        for script in [
            vec![1, 2, 3, 4, 5, 6, 14],
            vec![7, 8, 9, 10, 11, 14],
            vec![12, 14],
            vec![13, 14],
        ] {
            let text = generate_scripted_json(&script, 14);
            assert!(JsonConstraint::is_complete(&text), "{text}");
            assert!(
                serde_json::from_str::<serde_json::Value>(&text).is_ok(),
                "{text}"
            );
        }
    }

    fn grammar_token_texts() -> Vec<Option<String>> {
        vec![
            Some("x".to_string()),
            Some("{".to_string()),
            Some("\"name\"".to_string()),
            Some(":".to_string()),
            Some("\"bob\"".to_string()),
            Some(",".to_string()),
            Some("\"age\"".to_string()),
            Some("42".to_string()),
            Some("}".to_string()),
            Some(String::new()),
            Some("\"extra\"".to_string()),
            Some("true".to_string()),
            Some("A".to_string()),
            Some("B".to_string()),
            Some("Z".to_string()),
            Some("1".to_string()),
            Some("2".to_string()),
        ]
    }

    fn generate_scripted_grammar(
        kind: GrammarConstraintKind,
        grammar: &str,
        script: &[TokenId],
        eos_token_id: TokenId,
    ) -> anyhow::Result<String> {
        let token_texts = grammar_token_texts();
        let processor = ConstraintProcessor::new(
            Box::new(LlguidanceConstraint::from_token_texts(
                kind,
                grammar,
                &token_texts,
                Some(eos_token_id),
            )?),
            token_texts.clone(),
            vec![eos_token_id],
        );
        let mut generated_text = String::new();
        let mut generated_tokens = Vec::new();

        for (step, &desired) in script.iter().enumerate() {
            let context = ProcessorContext {
                generated_tokens: generated_tokens.clone(),
                generated_text: generated_text.clone(),
                step,
                ..Default::default()
            };
            let mut logits = vec![0.0; token_texts.len()];
            logits[0] = 100.0;
            logits[desired as usize] = 101.0;
            processor.process(&mut logits, &context).unwrap();
            assert!(
                logits[desired as usize].is_finite(),
                "desired token {desired} was masked at {generated_text:?}"
            );
            assert_eq!(logits[0], f32::NEG_INFINITY);
            let selected = sample_greedy(&logits);
            assert_eq!(selected, desired);
            if selected == eos_token_id {
                break;
            }
            generated_tokens.push(selected);
            generated_text.push_str(token_texts[selected as usize].as_deref().unwrap());
        }

        Ok(generated_text)
    }

    #[test]
    fn json_schema_constraint_generates_schema_valid_objects() -> anyhow::Result<()> {
        let schema = r#"{
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "age": { "type": "integer" }
            },
            "required": ["name", "age"],
            "additionalProperties": false
        }"#;

        for _ in 0..4 {
            let text = generate_scripted_grammar(
                GrammarConstraintKind::JsonSchema,
                schema,
                &[1, 2, 3, 4, 5, 6, 3, 7, 8, 9],
                9,
            )?;
            let value: serde_json::Value = serde_json::from_str(&text)?;
            assert_eq!(value["name"].as_str(), Some("bob"));
            assert_eq!(value["age"].as_i64(), Some(42));
            assert_eq!(value.as_object().map(|object| object.len()), Some(2));
        }

        Ok(())
    }

    #[test]
    fn regex_constraint_forces_matching_output() -> anyhow::Result<()> {
        let text = generate_scripted_grammar(
            GrammarConstraintKind::Regex,
            "[A-Z]{2}[0-9]{2}",
            &[12, 13, 15, 16, 9],
            9,
        )?;
        assert_eq!(text, "AB12");
        Ok(())
    }

    #[test]
    fn unconstrained_logits_are_unaffected() {
        let context = ProcessorContext::default();
        let mut logits = vec![10.0, 1.0];
        assert_eq!(sample_greedy(&logits), 0);

        let chain = ProcessorChain::new();
        chain.process(&mut logits, &context).unwrap();

        assert_eq!(sample_greedy(&logits), 0);
        assert_eq!(logits, vec![10.0, 1.0]);
    }

    struct ForceTokenProcessor {
        token_id: TokenId,
    }

    impl LogitProcessor for ForceTokenProcessor {
        fn process(&self, logits: &mut [f32], _context: &ProcessorContext) -> anyhow::Result<()> {
            for (idx, logit) in logits.iter_mut().enumerate() {
                *logit = if idx as TokenId == self.token_id {
                    0.0
                } else {
                    f32::NEG_INFINITY
                };
            }
            Ok(())
        }

        fn name(&self) -> &str {
            "force_token"
        }
    }

    struct EvenTokenConstraint;

    impl Constraint for EvenTokenConstraint {
        fn allowed_next_tokens(
            &self,
            _context: &ProcessorContext,
            candidates: &[TokenCandidate],
        ) -> anyhow::Result<Vec<bool>> {
            Ok(candidates
                .iter()
                .map(|candidate| candidate.token_id % 2 == 0)
                .collect())
        }

        fn name(&self) -> &str {
            "even_token_constraint"
        }
    }

    #[test]
    fn processor_chain_builder_registers_custom_processors_and_constraints() {
        let context = ProcessorContext::default();
        let chain = ProcessorChain::builder()
            .register_processor(Box::new(ForceTokenProcessor { token_id: 2 }))
            .register_constraint(
                Box::new(EvenTokenConstraint),
                vec![
                    Some("0".into()),
                    Some("1".into()),
                    Some("2".into()),
                    Some("3".into()),
                ],
                Vec::new(),
            )
            .build();
        let mut logits = vec![1.0, 1.0, 1.0, 1.0];

        chain.process(&mut logits, &context).unwrap();

        assert_eq!(
            logits,
            vec![f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY]
        );
        assert_eq!(chain.names(), vec!["force_token", "even_token_constraint"]);
    }
}

//! Logit processor chain.
//!
//! Processor order:
//! repetition/frequency/presence/DRY penalties -> constraints/stop checks ->
//! temperature -> top-k -> top-p -> min-p -> top-a -> typical-p ->
//! Mirostat -> XTC.

use std::collections::{HashMap, HashSet};
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
    fn allowed_next_tokens(
        &self,
        context: &ProcessorContext,
        candidates: &[TokenCandidate],
    ) -> Vec<bool>;

    fn name(&self) -> &str;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrammarConstraintKind {
    JsonSchema,
    Regex,
    Lark,
}

/// A logit processor modifies the logit distribution before sampling.
pub trait LogitProcessor: Send + Sync {
    fn process(&self, logits: &mut [f32], context: &ProcessorContext);
    fn name(&self) -> &str;

    /// Return a termination signal for the current context, if this processor owns one.
    fn signal(&self, _context: &ProcessorContext) -> Option<ProcessorSignal> {
        None
    }
}

/// Ordered chain of logit processors.
#[derive(Default)]
pub struct ProcessorChain {
    processors: Vec<Box<dyn LogitProcessor>>,
}

impl ProcessorChain {
    pub fn new() -> Self {
        Self {
            processors: Vec::new(),
        }
    }

    pub fn builder() -> ProcessorChainBuilder {
        ProcessorChainBuilder::new()
    }

    pub fn add(&mut self, processor: Box<dyn LogitProcessor>) {
        self.processors.push(processor);
    }

    /// Whether the chain has no processors. When empty, greedy decoding reduces
    /// to a plain argmax of the raw logits, enabling the decode fast path that
    /// selects the token on the device/buffer without materializing the full
    /// vocabulary on the host.
    pub fn is_empty(&self) -> bool {
        self.processors.is_empty()
    }

    pub fn add_constraint(
        &mut self,
        constraint: Box<dyn Constraint>,
        token_texts: Vec<Option<String>>,
        eos_token_id: Option<TokenId>,
    ) {
        self.add(Box::new(ConstraintProcessor::new(
            constraint,
            token_texts,
            eos_token_id,
        )));
    }

    /// Apply processors in insertion order.
    pub fn process(&self, logits: &mut [f32], context: &ProcessorContext) {
        for proc in &self.processors {
            proc.process(logits, context);
        }
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
        eos_token_id: Option<TokenId>,
    ) -> Self {
        self.chain
            .add_constraint(constraint, token_texts, eos_token_id);
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
    eos_token_id: Option<TokenId>,
}

impl ConstraintProcessor {
    pub fn new(
        constraint: Box<dyn Constraint>,
        token_texts: Vec<Option<String>>,
        eos_token_id: Option<TokenId>,
    ) -> Self {
        Self {
            constraint,
            token_texts,
            eos_token_id,
        }
    }
}

impl LogitProcessor for ConstraintProcessor {
    fn process(&self, logits: &mut [f32], context: &ProcessorContext) {
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
                    is_eos: self.eos_token_id == Some(token_id),
                }
            })
            .collect();
        let mask = self.constraint.allowed_next_tokens(context, &candidates);
        for (idx, logit) in logits.iter_mut().enumerate() {
            if !mask.get(idx).copied().unwrap_or(false) {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    fn name(&self) -> &str {
        self.constraint.name()
    }
}

/// Character-level JSON grammar constraint.
///
/// This validates the generated prefix plus each candidate token and only allows
/// tokens that keep the prefix valid and completable. EOS is allowed only after
/// a complete, balanced JSON value. JSON Schema constraints are future work.
#[derive(Debug, Clone, Default)]
pub struct JsonConstraint;

impl JsonConstraint {
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
    ) -> Vec<bool> {
        let complete = Self::is_complete(&context.generated_text);
        candidates
            .iter()
            .map(|candidate| {
                if candidate.is_eos {
                    return complete;
                }

                if candidate.text.is_empty() {
                    return false;
                }
                let mut next =
                    String::with_capacity(context.generated_text.len() + candidate.text.len());
                next.push_str(&context.generated_text);
                next.push_str(&candidate.text);
                Self::prefix_is_valid(&next)
            })
            .collect()
    }

    fn name(&self) -> &str {
        "json_constraint"
    }
}

/// llguidance-backed grammar constraint for JSON Schema, regex, and Lark grammars.
///
/// The state machine is advanced from `ProcessorContext::generated_tokens` before
/// computing the next-token mask. This keeps the existing logit-processor API
/// unchanged while still using llguidance's compute-mask/commit-token loop.
pub struct LlguidanceConstraint {
    kind: GrammarConstraintKind,
    inner: Mutex<LlguidanceState>,
}

struct LlguidanceState {
    constraint: LlgConstraint,
    committed_len: usize,
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
        Self::from_tok_env(kind, grammar, &tok_env)
    }

    pub fn from_token_texts(
        kind: GrammarConstraintKind,
        grammar: &str,
        token_texts: &[Option<String>],
        eos_token_id: Option<TokenId>,
    ) -> anyhow::Result<Self> {
        let mut token_bytes = Vec::with_capacity(token_texts.len());
        for (idx, text) in token_texts.iter().enumerate() {
            if Some(idx as TokenId) == eos_token_id {
                let mut bytes = b"<eos>".to_vec();
                bytes.insert(0, TokTrie::SPECIAL_TOKEN_MARKER);
                token_bytes.push(bytes);
            } else {
                token_bytes.push(text.as_deref().unwrap_or_default().as_bytes().to_vec());
            }
        }
        let info = TokRxInfo {
            vocab_size: token_bytes.len() as u32,
            tok_eos: eos_token_id.unwrap_or(0),
            tok_bos: None,
            tok_pad: None,
            tok_unk: None,
            tok_end_of_turn: None,
        };
        let tok_trie = TokTrie::from(&info, &token_bytes);
        let tok_env: llguidance::toktrie::TokEnv =
            std::sync::Arc::new(ApproximateTokEnv::new(tok_trie));
        Self::from_tok_env(kind, grammar, &tok_env)
    }

    fn from_tok_env(
        kind: GrammarConstraintKind,
        grammar: &str,
        tok_env: &llguidance::toktrie::TokEnv,
    ) -> anyhow::Result<Self> {
        let grammar = top_level_grammar(kind, grammar)?;
        let factory = ParserFactory::new(tok_env, InferenceCapabilities::default(), &[])?;
        let parser = factory.create_parser(grammar)?;
        Ok(Self {
            kind,
            inner: Mutex::new(LlguidanceState {
                constraint: LlgConstraint::new(parser),
                committed_len: 0,
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
    ) -> Vec<bool> {
        let mut inner = match self.inner.lock() {
            Ok(inner) => inner,
            Err(_) => return vec![false; candidates.len()],
        };

        if context.generated_tokens.len() < inner.committed_len {
            return vec![false; candidates.len()];
        }

        for token in &context.generated_tokens[inner.committed_len..] {
            if inner.constraint.commit_token(Some(*token)).is_err() {
                return vec![false; candidates.len()];
            }
            inner.committed_len += 1;
        }

        let step = match inner.constraint.compute_mask() {
            Ok(step) => step.clone(),
            Err(_) => return vec![false; candidates.len()],
        };

        if step.is_stop() {
            return candidates
                .iter()
                .map(|candidate| candidate.is_eos)
                .collect();
        }

        let Some(mask) = step.sample_mask.as_ref() else {
            return vec![false; candidates.len()];
        };

        candidates
            .iter()
            .map(|candidate| {
                let idx = candidate.token_id as usize;
                idx < mask.len() && mask.get(idx)
            })
            .collect()
    }

    fn name(&self) -> &str {
        match self.kind {
            GrammarConstraintKind::JsonSchema => "json_schema_constraint",
            GrammarConstraintKind::Regex => "regex_constraint",
            GrammarConstraintKind::Lark => "lark_constraint",
        }
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
}

impl LogitProcessor for TemperatureProcessor {
    fn process(&self, logits: &mut [f32], _context: &ProcessorContext) {
        if self.temperature.is_finite() && self.temperature > 0.0 && self.temperature != 1.0 {
            for logit in logits.iter_mut() {
                *logit /= self.temperature;
            }
        }
    }

    fn name(&self) -> &str {
        "temperature"
    }
}

pub struct RepetitionPenaltyProcessor {
    pub penalty: f32,
    /// When `Some(n)`, only the most recent `n` tokens of the combined
    /// prompt+generated stream are penalized; `None` penalizes the whole
    /// history (HF default behavior).
    pub window: Option<usize>,
}

impl LogitProcessor for RepetitionPenaltyProcessor {
    fn process(&self, logits: &mut [f32], context: &ProcessorContext) {
        if !self.penalty.is_finite() || self.penalty <= 0.0 || self.penalty == 1.0 {
            return;
        }

        let total = context.prompt_tokens.len() + context.generated_tokens.len();
        let skip = match self.window {
            Some(window) => total.saturating_sub(window),
            None => 0,
        };
        let mut seen = HashSet::new();
        for &token_id in context
            .prompt_tokens
            .iter()
            .chain(context.generated_tokens.iter())
            .skip(skip)
        {
            if !seen.insert(token_id) {
                continue;
            }
            if let Some(logit) = logits.get_mut(token_id as usize) {
                if *logit > 0.0 {
                    *logit /= self.penalty;
                } else {
                    *logit *= self.penalty;
                }
            }
        }
    }

    fn name(&self) -> &str {
        "repetition_penalty"
    }
}

pub struct FrequencyPenaltyProcessor {
    pub frequency_penalty: f32,
}

impl LogitProcessor for FrequencyPenaltyProcessor {
    fn process(&self, logits: &mut [f32], context: &ProcessorContext) {
        if !self.frequency_penalty.is_finite() || self.frequency_penalty == 0.0 {
            return;
        }

        let mut counts: HashMap<TokenId, usize> = HashMap::new();
        for &token_id in &context.generated_tokens {
            *counts.entry(token_id).or_default() += 1;
        }

        for (token_id, count) in counts {
            if let Some(logit) = logits.get_mut(token_id as usize) {
                *logit -= self.frequency_penalty * count as f32;
            }
        }
    }

    fn name(&self) -> &str {
        "frequency_penalty"
    }
}

pub struct PresencePenaltyProcessor {
    pub presence_penalty: f32,
}

impl LogitProcessor for PresencePenaltyProcessor {
    fn process(&self, logits: &mut [f32], context: &ProcessorContext) {
        if !self.presence_penalty.is_finite() || self.presence_penalty == 0.0 {
            return;
        }

        let mut seen = HashSet::new();
        for &token_id in &context.generated_tokens {
            if seen.insert(token_id)
                && let Some(logit) = logits.get_mut(token_id as usize)
            {
                *logit -= self.presence_penalty;
            }
        }
    }

    fn name(&self) -> &str {
        "presence_penalty"
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
    pub sequence_breakers: Vec<TokenId>,
}

impl LogitProcessor for DryProcessor {
    fn process(&self, logits: &mut [f32], context: &ProcessorContext) {
        if !self.multiplier.is_finite()
            || self.multiplier <= 0.0
            || !self.base.is_finite()
            || self.base < 1.0
            || self.allowed_length == 0
        {
            return;
        }

        let tokens: Vec<_> = context
            .prompt_tokens
            .iter()
            .chain(&context.generated_tokens)
            .copied()
            .collect();
        if tokens.len() <= self.allowed_length {
            return;
        }

        let breakers: HashSet<_> = self.sequence_breakers.iter().copied().collect();
        let mut penalties: HashMap<TokenId, f32> = HashMap::new();
        for continuation_index in 1..tokens.len() {
            let continuation = tokens[continuation_index];
            if breakers.contains(&continuation) {
                continue;
            }

            let mut matched_length = 0;
            let mut historical_index = continuation_index;
            let mut recent_index = tokens.len();
            while historical_index > 0 && recent_index > 0 {
                let historical = tokens[historical_index - 1];
                let recent = tokens[recent_index - 1];
                if historical != recent
                    || breakers.contains(&historical)
                    || breakers.contains(&recent)
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
    fn process(&self, _logits: &mut [f32], _context: &ProcessorContext) {}

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

pub struct TopKProcessor {
    pub top_k: usize,
}

impl LogitProcessor for TopKProcessor {
    fn process(&self, logits: &mut [f32], _context: &ProcessorContext) {
        if self.top_k == 0 || self.top_k >= logits.len() {
            return;
        }

        let mut sorted: Vec<f32> = logits.iter().copied().filter(|v| !v.is_nan()).collect();
        if sorted.is_empty() {
            return;
        }
        sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sorted[self.top_k.saturating_sub(1).min(sorted.len() - 1)];

        for logit in logits.iter_mut() {
            if logit.is_nan() || *logit < threshold {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    fn name(&self) -> &str {
        "top_k"
    }
}

pub struct TopPProcessor {
    pub top_p: f32,
}

impl LogitProcessor for TopPProcessor {
    fn process(&self, logits: &mut [f32], _context: &ProcessorContext) {
        if !self.top_p.is_finite() || self.top_p >= 1.0 || logits.is_empty() {
            return;
        }

        let Some(probabilities) = softmax_probabilities(logits) else {
            return;
        };
        let probs = ranked_probabilities(&probabilities);

        let mut cumulative = 0.0;
        let mut keep_count = 0;
        let cutoff = self.top_p.max(0.0);
        for &(_, prob) in &probs {
            keep_count += 1;
            cumulative += prob;
            if cumulative >= cutoff {
                break;
            }
        }

        for &(idx, _) in probs.iter().skip(keep_count) {
            logits[idx] = f32::NEG_INFINITY;
        }
    }

    fn name(&self) -> &str {
        "top_p"
    }
}

pub struct MinPProcessor {
    pub min_p: f32,
}

impl LogitProcessor for MinPProcessor {
    fn process(&self, logits: &mut [f32], _context: &ProcessorContext) {
        if !self.min_p.is_finite() || self.min_p <= 0.0 || logits.is_empty() {
            return;
        }

        let Some(probabilities) = softmax_probabilities(logits) else {
            return;
        };

        let top_prob = probabilities.iter().copied().fold(0.0, f32::max);
        let threshold = self.min_p.min(1.0) * top_prob;
        for (logit, prob) in logits.iter_mut().zip(probabilities) {
            if !prob.is_finite() || prob < threshold {
                *logit = f32::NEG_INFINITY;
            }
        }
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
    fn process(&self, logits: &mut [f32], _context: &ProcessorContext) {
        if !self.top_a.is_finite() || self.top_a <= 0.0 || logits.is_empty() {
            return;
        }
        let Some(probabilities) = softmax_probabilities(logits) else {
            return;
        };
        let maximum_probability = probabilities.iter().copied().fold(0.0, f32::max);
        let threshold = self.top_a.min(1.0) * maximum_probability * maximum_probability;
        for (logit, probability) in logits.iter_mut().zip(probabilities) {
            if probability < threshold {
                *logit = f32::NEG_INFINITY;
            }
        }
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
    fn process(&self, logits: &mut [f32], _context: &ProcessorContext) {
        if !self.typical_p.is_finite() || self.typical_p >= 1.0 || logits.is_empty() {
            return;
        }
        let Some(probabilities) = softmax_probabilities(logits) else {
            return;
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
    fn process(&self, logits: &mut [f32], context: &ProcessorContext) {
        if !self.tau.is_finite()
            || self.tau <= 0.0
            || !self.eta.is_finite()
            || self.eta <= 0.0
            || logits.is_empty()
        {
            return;
        }
        let Some(probabilities) = softmax_probabilities(logits) else {
            return;
        };
        let mut state = match self.state.lock() {
            Ok(state) => state,
            Err(_) => return,
        };

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
    fn process(&self, logits: &mut [f32], _context: &ProcessorContext) {
        if !self.probability.is_finite()
            || self.probability <= 0.0
            || !self.threshold.is_finite()
            || logits.is_empty()
        {
            return;
        }
        let should_apply = match self.rng.lock() {
            Ok(mut rng) => rng.random::<f32>() < self.probability.min(1.0),
            Err(_) => false,
        };
        if !should_apply {
            return;
        }
        let Some(probabilities) = softmax_probabilities(logits) else {
            return;
        };
        let candidates: Vec<_> = probabilities
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, probability)| *probability >= self.threshold.max(0.0))
            .collect();
        if candidates.len() < 2 {
            return;
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

fn ranked_probabilities(probabilities: &[f32]) -> Vec<(usize, f32)> {
    let mut ranked: Vec<_> = probabilities.iter().copied().enumerate().collect();
    ranked.sort_unstable_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    ranked
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::sample_greedy;

    fn context(prompt_tokens: Vec<TokenId>, generated_tokens: Vec<TokenId>) -> ProcessorContext {
        ProcessorContext {
            prompt_tokens,
            generated_tokens,
            generated_text: String::new(),
            step: 0,
        }
    }

    #[test]
    fn repetition_penalty_applies_once_per_seen_token() {
        let processor = RepetitionPenaltyProcessor {
            penalty: 2.0,
            window: None,
        };
        let mut logits = vec![4.0, -4.0, 8.0];
        processor.process(&mut logits, &context(vec![0, 0], vec![1, 1]));
        assert_eq!(logits, vec![2.0, -8.0, 8.0]);
    }

    #[test]
    fn repetition_penalty_window_limits_to_recent_tokens() {
        // Combined stream is prompt [0,1] ++ generated [2,3]; a window of 2 keeps
        // only the last two tokens (2 and 3), leaving 0 and 1 untouched.
        let processor = RepetitionPenaltyProcessor {
            penalty: 2.0,
            window: Some(2),
        };
        let mut logits = vec![4.0, 4.0, 4.0, -4.0];
        processor.process(&mut logits, &context(vec![0, 1], vec![2, 3]));
        assert_eq!(logits, vec![4.0, 4.0, 2.0, -8.0]);
    }

    #[test]
    fn repetition_penalty_window_zero_penalizes_nothing() {
        let processor = RepetitionPenaltyProcessor {
            penalty: 2.0,
            window: Some(0),
        };
        let mut logits = vec![4.0, 4.0];
        processor.process(&mut logits, &context(vec![0], vec![1]));
        assert_eq!(logits, vec![4.0, 4.0]);
    }

    #[test]
    fn repetition_penalty_disabled_is_identity() {
        let processor = RepetitionPenaltyProcessor {
            penalty: 1.0,
            window: None,
        };
        let mut logits = vec![4.0, -4.0, 8.0];
        let original = logits.clone();
        processor.process(&mut logits, &context(vec![0, 1], vec![2]));
        assert_eq!(logits, original);
    }

    #[test]
    fn frequency_penalty_scales_with_generated_count() {
        let processor = FrequencyPenaltyProcessor {
            frequency_penalty: 0.5,
        };
        let mut logits = vec![4.0, 4.0, 4.0];
        processor.process(&mut logits, &context(vec![0, 0], vec![0, 1, 1]));
        assert_eq!(logits, vec![3.5, 3.0, 4.0]);
    }

    #[test]
    fn presence_penalty_applies_once_per_generated_token() {
        let processor = PresencePenaltyProcessor {
            presence_penalty: 0.75,
        };
        let mut logits = vec![4.0, 4.0, 4.0];
        processor.process(&mut logits, &context(vec![0], vec![1, 1, 2]));
        assert_eq!(logits, vec![4.0, 3.25, 3.25]);
    }

    #[test]
    fn top_k_masks_tokens_below_threshold() {
        let processor = TopKProcessor { top_k: 2 };
        let mut logits = vec![0.0, 5.0, 1.0, 4.0];
        processor.process(&mut logits, &ProcessorContext::default());
        assert_eq!(logits, vec![f32::NEG_INFINITY, 5.0, f32::NEG_INFINITY, 4.0]);
    }

    #[test]
    fn top_p_keeps_minimal_nucleus_and_at_least_one_token() {
        let processor = TopPProcessor { top_p: 0.6 };
        let mut logits = vec![3.0, 2.0, 1.0];
        processor.process(&mut logits, &ProcessorContext::default());
        assert!(logits[0].is_finite());
        assert_eq!(logits[1], f32::NEG_INFINITY);
        assert_eq!(logits[2], f32::NEG_INFINITY);
    }

    #[test]
    fn min_p_masks_relative_to_top_token_probability() {
        let processor = MinPProcessor { min_p: 0.5 };
        let mut logits = vec![0.0, -0.5, -1.0];
        processor.process(&mut logits, &ProcessorContext::default());
        assert!(logits[0].is_finite());
        assert!(logits[1].is_finite());
        assert_eq!(logits[2], f32::NEG_INFINITY);
    }

    #[test]
    fn top_a_uses_squared_top_probability_threshold_inclusively() {
        let processor = TopAProcessor { top_a: 0.5 };
        let mut logits = vec![0.6_f32.ln(), 0.3_f32.ln(), 0.1_f32.ln()];
        processor.process(&mut logits, &ProcessorContext::default());
        assert!(logits[0].is_finite());
        assert!(logits[1].is_finite());
        assert_eq!(logits[2], f32::NEG_INFINITY);

        let mut boundary = vec![0.5_f32.ln(), 0.25_f32.ln(), 0.25_f32.ln()];
        TopAProcessor { top_a: 1.0 }.process(&mut boundary, &ProcessorContext::default());
        assert!(boundary.iter().all(|logit| logit.is_finite()));
    }

    #[test]
    fn top_a_disabled_equal_and_single_token_distributions_are_safe() {
        let original = vec![0.0, 0.0, 0.0];
        let mut disabled = original.clone();
        TopAProcessor { top_a: 0.0 }.process(&mut disabled, &ProcessorContext::default());
        assert_eq!(disabled, original);

        let mut equal = vec![0.0, 0.0, 0.0, 0.0];
        TopAProcessor { top_a: 1.0 }.process(&mut equal, &ProcessorContext::default());
        assert!(equal.iter().all(|logit| logit.is_finite()));

        let mut single = vec![4.0];
        TopAProcessor { top_a: 1.0 }.process(&mut single, &ProcessorContext::default());
        assert_eq!(single, vec![4.0]);
    }

    #[test]
    fn typical_p_keeps_tokens_nearest_entropy_until_mass_is_reached() {
        // For [0.5, 0.3, 0.2], token 1 is most typical, followed by 0, then 2.
        let processor = TypicalPProcessor { typical_p: 0.6 };
        let mut logits = vec![0.5_f32.ln(), 0.3_f32.ln(), 0.2_f32.ln()];
        processor.process(&mut logits, &ProcessorContext::default());
        assert!(logits[0].is_finite());
        assert!(logits[1].is_finite());
        assert_eq!(logits[2], f32::NEG_INFINITY);
    }

    #[test]
    fn typical_p_handles_noop_equal_and_single_token_distributions() {
        let original = vec![1.0, 0.0];
        let mut disabled = original.clone();
        TypicalPProcessor { typical_p: 1.0 }.process(&mut disabled, &ProcessorContext::default());
        assert_eq!(disabled, original);

        let mut equal = vec![0.0; 4];
        TypicalPProcessor { typical_p: 0.5 }.process(&mut equal, &ProcessorContext::default());
        assert!(equal[0].is_finite());
        assert!(equal[1].is_finite());
        assert_eq!(equal[2], f32::NEG_INFINITY);
        assert_eq!(equal[3], f32::NEG_INFINITY);

        let mut single = vec![0.0];
        TypicalPProcessor { typical_p: 0.1 }.process(&mut single, &ProcessorContext::default());
        assert_eq!(single, vec![0.0]);
    }

    #[test]
    fn mirostat_v1_estimates_zipf_slope_and_keeps_expected_top_k() {
        // p(rank) proportional to rank^-2 gives slope=2. With mu=2 and n=4,
        // k=round(sqrt(4 / (1 - 4^-1)))=2.
        let processor = MirostatProcessor::new(1.0, 0.1, MirostatVersion::V1);
        let mut logits = vec![0.0, -(4.0_f32).ln(), -(9.0_f32).ln(), -(16.0_f32).ln()];
        processor.process(&mut logits, &ProcessorContext::default());
        assert!(logits[0].is_finite());
        assert!(logits[1].is_finite());
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[3], f32::NEG_INFINITY);
    }

    #[test]
    fn mirostat_v2_cutoff_and_feedback_move_mu_toward_target_surprise() {
        let processor = MirostatProcessor::new(0.5, 0.25, MirostatVersion::V2);
        let mut first = vec![0.6_f32.ln(), 0.3_f32.ln(), 0.1_f32.ln()];
        processor.process(&mut first, &ProcessorContext::default());
        assert!(first[0].is_finite());
        assert_eq!(first[1], f32::NEG_INFINITY);
        assert_eq!(first[2], f32::NEG_INFINITY);

        let initial_mu = processor.current_mu();
        let mut second = vec![0.9_f32.ln(), 0.1_f32.ln()];
        processor.process(
            &mut second,
            &ProcessorContext {
                generated_tokens: vec![1],
                ..Default::default()
            },
        );
        assert!(
            processor.current_mu() < initial_mu,
            "a token more surprising than tau must lower mu"
        );

        let lowered_mu = processor.current_mu();
        let mut third = vec![0.9_f32.ln(), 0.1_f32.ln()];
        processor.process(
            &mut third,
            &ProcessorContext {
                generated_tokens: vec![1, 0],
                ..Default::default()
            },
        );
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
            processor.process(&mut logits, &ProcessorContext::default());
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
        };
        let mut logits = vec![0.0; 5];
        processor.process(&mut logits, &context(vec![1, 2, 3], vec![1, 2]));
        assert_eq!(logits[3], -2.0);
        assert_eq!(logits[0], 0.0);
        assert_eq!(logits[1], 0.0);
        assert_eq!(logits[2], 0.0);
        assert_eq!(logits[4], 0.0);
    }

    #[test]
    fn dry_penalty_grows_and_sequence_breakers_stop_matching() {
        let mut logits = vec![0.0; 10];
        DryProcessor {
            multiplier: 2.0,
            base: 1.5,
            allowed_length: 2,
            sequence_breakers: vec![],
        }
        .process(&mut logits, &context(vec![1, 2, 3, 4], vec![1, 2, 3]));
        assert_eq!(logits[4], -3.0);

        let mut broken = vec![0.0; 10];
        DryProcessor {
            multiplier: 2.0,
            base: 1.5,
            allowed_length: 2,
            sequence_breakers: vec![9],
        }
        .process(&mut broken, &context(vec![1, 9, 2, 3], vec![1, 9, 2]));
        assert!(broken.iter().all(|logit| *logit == 0.0));
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
        }
        .process(&mut logits, &context(vec![1, 2, 3], vec![1, 2]));
        assert_eq!(logits, original);
    }

    #[test]
    fn xtc_probability_one_excludes_top_choices_except_least_probable() {
        let processor = XtcProcessor::new(1.0, 0.25, Some(7));
        let mut logits = vec![0.5_f32.ln(), 0.3_f32.ln(), 0.2_f32.ln()];
        processor.process(&mut logits, &ProcessorContext::default());
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert!(logits[1].is_finite());
        assert!(logits[2].is_finite());
    }

    #[test]
    fn xtc_probability_zero_is_noop_and_single_token_is_safe() {
        let original = vec![2.0, 1.0];
        let mut disabled = original.clone();
        XtcProcessor::new(0.0, 0.1, Some(7)).process(&mut disabled, &ProcessorContext::default());
        assert_eq!(disabled, original);

        let mut single = vec![2.0];
        XtcProcessor::new(1.0, 0.0, Some(7)).process(&mut single, &ProcessorContext::default());
        assert_eq!(single, vec![2.0]);
    }

    #[test]
    fn xtc_uses_seeded_reproducible_probability_draws() {
        let left = XtcProcessor::new(0.5, 0.2, Some(99));
        let right = XtcProcessor::new(0.5, 0.2, Some(99));
        let run = |processor: &XtcProcessor| {
            (0..32)
                .map(|_| {
                    let mut logits = vec![0.5_f32.ln(), 0.3_f32.ln(), 0.2_f32.ln()];
                    processor.process(&mut logits, &ProcessorContext::default());
                    logits
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(run(&left), run(&right));
    }

    #[test]
    fn temperature_scales_logits() {
        let processor = TemperatureProcessor { temperature: 2.0 };
        let mut logits = vec![2.0, -4.0];
        processor.process(&mut logits, &ProcessorContext::default());
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
            Box::new(JsonConstraint),
            json_token_texts(),
            Some(eos_token_id),
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
            processor.process(&mut logits, &context);
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
    fn json_constraint_masks_invalid_tokens_and_allows_eos_only_when_complete() {
        let processor =
            ConstraintProcessor::new(Box::new(JsonConstraint), json_token_texts(), Some(14));

        let mut logits = vec![0.0; json_token_texts().len()];
        logits[6] = 10.0;
        logits[1] = 1.0;
        processor.process(&mut logits, &ProcessorContext::default());
        assert_eq!(logits[6], f32::NEG_INFINITY);
        assert!(logits[1].is_finite());
        assert_eq!(logits[14], f32::NEG_INFINITY);

        let complete_context = ProcessorContext {
            generated_text: "{\"a\":1}".to_string(),
            ..Default::default()
        };
        let mut complete_logits = vec![0.0; json_token_texts().len()];
        processor.process(&mut complete_logits, &complete_context);
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
            Some(eos_token_id),
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
            processor.process(&mut logits, &context);
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
        chain.process(&mut logits, &context);

        assert_eq!(sample_greedy(&logits), 0);
        assert_eq!(logits, vec![10.0, 1.0]);
    }

    struct ForceTokenProcessor {
        token_id: TokenId,
    }

    impl LogitProcessor for ForceTokenProcessor {
        fn process(&self, logits: &mut [f32], _context: &ProcessorContext) {
            for (idx, logit) in logits.iter_mut().enumerate() {
                *logit = if idx as TokenId == self.token_id {
                    0.0
                } else {
                    f32::NEG_INFINITY
                };
            }
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
        ) -> Vec<bool> {
            candidates
                .iter()
                .map(|candidate| candidate.token_id % 2 == 0)
                .collect()
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
                None,
            )
            .build();
        let mut logits = vec![1.0, 1.0, 1.0, 1.0];

        chain.process(&mut logits, &context);

        assert_eq!(
            logits,
            vec![f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY]
        );
        assert_eq!(chain.names(), vec!["force_token", "even_token_constraint"]);
    }
}

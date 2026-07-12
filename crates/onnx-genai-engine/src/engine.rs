//! Main generation engine.

use crate::logits::{
    ProcessorChain, ProcessorContext, ProcessorSignal, RepetitionPenaltyProcessor, StopSequence,
    StopSequenceProcessor, TemperatureProcessor, TokenId, TopKProcessor, TopPProcessor,
};
use crate::sampling::{sample_categorical, sample_greedy};
use anyhow::Context;
use onnx_genai_kv::{PagedKvCache, SequenceId};
use onnx_genai_metadata::InferenceMetadata;
use onnx_genai_ort::{
    DataType, Environment, ModelDirectory, Session, SessionOptions, TensorInfo, Tokenizer, Value,
};
use onnx_genai_scheduler::{Priority, Scheduler, SchedulerConfig};
use std::collections::HashMap;
use std::path::Path;

/// Engine configuration.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Number of GPU pages for KV cache.
    pub num_gpu_pages: usize,
    /// Tokens per KV page.
    pub page_size: usize,
    /// Scheduler config.
    pub scheduler: SchedulerConfig,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            num_gpu_pages: 1024,
            page_size: 16,
            scheduler: SchedulerConfig::default(),
        }
    }
}

/// Prompt input accepted by Phase 1 generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeneratePrompt {
    /// Raw prompt text.
    Text(String),
    /// Already-tokenized prompt ids.
    TokenIds(Vec<TokenId>),
}

impl From<String> for GeneratePrompt {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for GeneratePrompt {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl From<Vec<TokenId>> for GeneratePrompt {
    fn from(value: Vec<TokenId>) -> Self {
        Self::TokenIds(value)
    }
}

/// User-controllable decoding options for Phase 1 generation.
#[derive(Debug, Clone)]
pub struct GenerateOptions {
    /// Maximum tokens to produce after the prompt.
    pub max_new_tokens: usize,
    /// Temperature applied before sampling. Zero forces greedy selection.
    pub temperature: f32,
    /// Nucleus sampling probability. Values >= 1 disable top-p filtering.
    pub top_p: f32,
    /// Keep only the top-k logits before sampling. Zero disables top-k filtering.
    pub top_k: usize,
    /// Repetition penalty applied to prompt and generated tokens. Values <= 1 disable it.
    pub repetition_penalty: f32,
    /// If true, choose argmax after processors; otherwise sample categorically.
    pub greedy: bool,
    /// Text or token sequences that terminate generation when matched as a suffix.
    pub stop_sequences: Vec<StopSequence>,
    /// Optional EOS token id.
    pub eos_token_id: Option<TokenId>,
    /// Whether matching `eos_token_id` terminates generation.
    pub stop_on_eos: bool,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            greedy: true,
            stop_sequences: Vec::new(),
            eos_token_id: None,
            stop_on_eos: true,
        }
    }
}

impl GenerateOptions {
    fn validate(&self) -> anyhow::Result<()> {
        if self.max_new_tokens == 0 {
            anyhow::bail!("max_new_tokens must be greater than zero");
        }
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            anyhow::bail!("temperature must be finite and non-negative");
        }
        if !self.top_p.is_finite() || self.top_p < 0.0 {
            anyhow::bail!("top_p must be finite and non-negative");
        }
        if !self.repetition_penalty.is_finite() || self.repetition_penalty <= 0.0 {
            anyhow::bail!("repetition_penalty must be finite and greater than zero");
        }
        Ok(())
    }
}

/// A single generation request.
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    /// Prompt text or token ids.
    pub prompt: GeneratePrompt,
    /// Decoding options.
    pub options: GenerateOptions,
}

impl GenerateRequest {
    pub fn new(prompt: impl Into<GeneratePrompt>) -> Self {
        Self {
            prompt: prompt.into(),
            options: GenerateOptions::default(),
        }
    }
}

/// Why generation stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    /// The configured maximum number of new tokens was reached.
    MaxTokens,
    /// The configured EOS token was generated.
    EosToken,
    /// A stop sequence matched; index refers to `GenerateOptions::stop_sequences`.
    StopSequence { index: usize },
}

/// Final generation output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerateResult {
    /// Detokenized generated text.
    pub text: String,
    /// Generated token ids, excluding prompt tokens.
    pub token_ids: Vec<TokenId>,
    /// Termination reason.
    pub finish_reason: FinishReason,
}

/// Per-token streaming event shape for future callback/iterator APIs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerateToken {
    pub token_id: TokenId,
    pub text: String,
    pub finish_reason: Option<FinishReason>,
}

/// Streaming callback shape. Returning an error aborts generation.
pub type GenerateTokenCallback<'a> = dyn FnMut(GenerateToken) -> anyhow::Result<()> + Send + 'a;

/// The generation engine.
pub struct Engine {
    /// Model inference metadata.
    metadata: InferenceMetadata,
    /// KV cache manager.
    kv_cache: PagedKvCache,
    /// Batch scheduler.
    scheduler: Scheduler,
    /// ORT environment kept alive for the session.
    _environment: Environment,
    /// ORT session for decoder execution.
    session: Session,
    /// Tokenizer loaded from the model directory.
    tokenizer: Tokenizer,
}

impl Engine {
    /// Load a model from a directory.
    pub fn from_dir(model_dir: &Path, config: EngineConfig) -> anyhow::Result<Self> {
        let model_directory = ModelDirectory::load(model_dir)
            .map_err(|e| anyhow::anyhow!("Failed to resolve model directory: {}", e))?;

        // Load metadata
        let metadata = if let Some(metadata_path) = &model_directory.metadata_path {
            onnx_genai_metadata::load_metadata(metadata_path)
                .map_err(|e| anyhow::anyhow!("Failed to load metadata: {}", e))?
        } else {
            tracing::warn!("No inference metadata found, using defaults");
            InferenceMetadata {
                required_capabilities: vec![],
                model: None,
                kv_cache: None,
                quantization: None,
                pipeline: None,
                strategy: None,
                structured_output: None,
                hardware_requirements: None,
            }
        };

        // Validate capabilities
        let runtime_caps = onnx_genai_metadata::RuntimeCapabilities::default();
        if let Err(unsupported) = onnx_genai_metadata::validate(&metadata, &runtime_caps) {
            anyhow::bail!("Unsupported capabilities: {:?}", unsupported);
        }

        // Initialize KV cache
        let kv_cache = PagedKvCache::new(config.page_size, config.num_gpu_pages);

        // Initialize scheduler
        let scheduler = Scheduler::new(config.scheduler);

        let environment = Environment::new("onnx-genai-engine")
            .map_err(|e| anyhow::anyhow!("Failed to create ORT environment: {}", e))?;
        let session = Session::new(
            &environment,
            &model_directory.model_path,
            SessionOptions::default(),
        )
        .map_err(|e| anyhow::anyhow!("Failed to load ORT session: {}", e))?;
        let tokenizer = Tokenizer::from_file(&model_directory.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(Self {
            metadata,
            kv_cache,
            scheduler,
            _environment: environment,
            session,
            tokenizer,
        })
    }

    /// Generate text for a request.
    ///
    pub fn generate(&mut self, request: GenerateRequest) -> anyhow::Result<GenerateResult> {
        self.generate_with_callback(request, None)
    }

    /// Generate text and optionally stream each generated token to `callback`.
    pub fn generate_with_callback(
        &mut self,
        request: GenerateRequest,
        mut callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        request.options.validate()?;
        let mut options = request.options.clone();
        if options.eos_token_id.is_none() {
            options.eos_token_id = self.tokenizer.eos_token_id();
        }
        let prompt_tokens = self.tokenize_prompt(&request.prompt)?;
        if prompt_tokens.is_empty() {
            anyhow::bail!("prompt must contain at least one token");
        }
        let mut decode_state = DecodeState::new(&self.session)?;
        let seq_id = self.create_session();
        self.scheduler.add_request(
            seq_id,
            prompt_tokens.len(),
            options.max_new_tokens,
            Priority::Normal,
        );

        let chain = build_processor_chain(&options);
        let mut generated_tokens = Vec::new();
        let mut generated_text = String::new();

        for step in 0..options.max_new_tokens {
            let mut context = ProcessorContext {
                prompt_tokens: prompt_tokens.clone(),
                generated_tokens: generated_tokens.clone(),
                generated_text: generated_text.clone(),
                step,
            };

            let mut logits = self.next_token_logits(
                seq_id,
                &prompt_tokens,
                &generated_tokens,
                &mut decode_state,
            )?;
            let token_id = select_next_token(&mut logits, &context, &options, &chain, 0.0);
            generated_tokens.push(token_id);
            self.scheduler.advance(seq_id);

            let token_text = self.detokenize_token(token_id)?;
            generated_text.push_str(&token_text);
            context.generated_tokens = generated_tokens.clone();
            context.generated_text = generated_text.clone();

            let finish_reason = finish_reason_after_token(token_id, &options, &chain, &context);
            if let Some(callback) = callback.as_deref_mut() {
                callback(GenerateToken {
                    token_id,
                    text: token_text,
                    finish_reason: finish_reason.clone(),
                })?;
            }

            if let Some(finish_reason) = finish_reason {
                self.scheduler.complete(seq_id);
                return Ok(GenerateResult {
                    text: self.detokenize_tokens(&generated_tokens)?,
                    token_ids: generated_tokens,
                    finish_reason,
                });
            }
        }

        self.scheduler.complete(seq_id);
        Ok(GenerateResult {
            text: self.detokenize_tokens(&generated_tokens)?,
            token_ids: generated_tokens,
            finish_reason: FinishReason::MaxTokens,
        })
    }

    /// Create a new generation session.
    pub fn create_session(&mut self) -> SequenceId {
        self.kv_cache.create_sequence()
    }

    /// Get the loaded metadata.
    pub fn metadata(&self) -> &InferenceMetadata {
        &self.metadata
    }

    fn tokenize_prompt(&self, prompt: &GeneratePrompt) -> anyhow::Result<Vec<TokenId>> {
        match prompt {
            GeneratePrompt::TokenIds(tokens) => Ok(tokens.clone()),
            GeneratePrompt::Text(text) => self
                .tokenizer
                .encode(text)
                .map_err(|e| anyhow::anyhow!("Failed to tokenize prompt: {}", e)),
        }
    }

    fn detokenize_token(&self, token_id: TokenId) -> anyhow::Result<String> {
        self.tokenizer
            .decode(&[token_id])
            .map_err(|e| anyhow::anyhow!("Failed to detokenize token {token_id}: {}", e))
    }

    fn detokenize_tokens(&self, token_ids: &[TokenId]) -> anyhow::Result<String> {
        self.tokenizer
            .decode(token_ids)
            .map_err(|e| anyhow::anyhow!("Failed to detokenize generated tokens: {}", e))
    }

    fn next_token_logits(
        &mut self,
        _seq_id: SequenceId,
        prompt_tokens: &[TokenId],
        generated_tokens: &[TokenId],
        decode_state: &mut DecodeState,
    ) -> anyhow::Result<Vec<f32>> {
        let (input_tokens, past_len) =
            decode_input_tokens(prompt_tokens, generated_tokens, decode_state)?;
        let outputs = run_decode_step(&self.session, decode_state, &input_tokens, past_len)?;
        extract_next_token_logits(&self.session, outputs)
    }
}

struct DecodeState {
    use_kv: bool,
    past: HashMap<String, Value>,
    present_to_past: HashMap<String, String>,
    kv_inputs: Vec<String>,
}

impl DecodeState {
    fn new(session: &Session) -> anyhow::Result<Self> {
        let kv_inputs = session
            .inputs()
            .iter()
            .filter(|info| is_kv_input(&info.name))
            .map(|info| info.name.clone())
            .collect::<Vec<_>>();
        let present_outputs = session
            .outputs()
            .iter()
            .filter(|info| is_present_output(&info.name))
            .map(|info| info.name.clone())
            .collect::<Vec<_>>();

        if kv_inputs.is_empty() && present_outputs.is_empty() {
            return Ok(Self {
                use_kv: false,
                past: HashMap::new(),
                present_to_past: HashMap::new(),
                kv_inputs,
            });
        }

        let mut present_to_past = HashMap::new();
        for output in &present_outputs {
            if let Some(input) = matching_past_input(output, &kv_inputs) {
                present_to_past.insert(output.clone(), input.clone());
            }
        }

        if kv_inputs.is_empty()
            || present_outputs.is_empty()
            || present_to_past.len() != present_outputs.len()
        {
            anyhow::bail!(
                "model exposes incomplete KV I/O; past inputs: {:?}, present outputs: {:?}",
                kv_inputs,
                present_outputs
            );
        }

        Ok(Self {
            use_kv: true,
            past: HashMap::new(),
            present_to_past,
            kv_inputs,
        })
    }
}

fn decode_input_tokens(
    prompt_tokens: &[TokenId],
    generated_tokens: &[TokenId],
    decode_state: &DecodeState,
) -> anyhow::Result<(Vec<TokenId>, usize)> {
    if decode_state.use_kv && !decode_state.past.is_empty() {
        let token = generated_tokens
            .last()
            .copied()
            .context("KV decode step has no generated token to feed")?;
        Ok((
            vec![token],
            prompt_tokens.len() + generated_tokens.len() - 1,
        ))
    } else {
        let mut tokens = prompt_tokens.to_vec();
        if !decode_state.use_kv {
            tokens.extend_from_slice(generated_tokens);
        }
        Ok((tokens, 0))
    }
}

fn run_decode_step(
    session: &Session,
    decode_state: &mut DecodeState,
    token_ids: &[TokenId],
    past_len: usize,
) -> anyhow::Result<Vec<Value>> {
    if token_ids.is_empty() {
        anyhow::bail!("decode step requires at least one input token");
    }

    let seq_len = token_ids.len();
    let total_len = past_len + seq_len;
    let input_ids = token_ids
        .iter()
        .map(|&id| i64::from(id))
        .collect::<Vec<_>>();
    let attention_mask = vec![1_i64; total_len];
    let position_ids = (past_len..total_len)
        .map(|pos| i64::try_from(pos).context("position id exceeds i64 range"))
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut owned_inputs: Vec<(String, Value)> = Vec::new();
    for info in session.inputs() {
        let lower = info.name.to_ascii_lowercase();
        if lower == "input_ids" || lower.ends_with(".input_ids") {
            ensure_i64(info)?;
            owned_inputs.push((
                info.name.clone(),
                Value::from_slice_i64(&input_ids, &[1, seq_len as i64])?,
            ));
        } else if lower == "attention_mask" || lower.ends_with(".attention_mask") {
            ensure_i64(info)?;
            owned_inputs.push((
                info.name.clone(),
                Value::from_slice_i64(&attention_mask, &[1, total_len as i64])?,
            ));
        } else if lower == "position_ids" || lower.ends_with(".position_ids") {
            ensure_i64(info)?;
            owned_inputs.push((
                info.name.clone(),
                Value::from_slice_i64(&position_ids, &[1, seq_len as i64])?,
            ));
        } else if decode_state.use_kv && decode_state.kv_inputs.contains(&info.name) {
            let value = if past_len == 0 {
                empty_past_value(info)?
            } else {
                decode_state.past.remove(&info.name).with_context(|| {
                    format!("missing cached KV tensor for input '{}'", info.name)
                })?
            };
            owned_inputs.push((info.name.clone(), value));
        } else {
            anyhow::bail!(
                "unsupported model input '{}' with shape {:?}; supported inputs are input_ids, attention_mask, position_ids, and past key-values",
                info.name,
                info.shape
            );
        }
    }

    let input_refs = owned_inputs
        .iter()
        .map(|(name, value)| (name.as_str(), value))
        .collect::<Vec<_>>();
    let outputs = session
        .run(&input_refs)
        .map_err(|e| anyhow::anyhow!("ORT session run failed: {}", e))?;

    if decode_state.use_kv {
        decode_state.past.clear();
        for (name, value) in session.output_names().iter().zip(outputs.iter()) {
            if let Some(past_name) = decode_state.present_to_past.get(name) {
                decode_state
                    .past
                    .insert(past_name.clone(), clone_value(value)?);
            }
        }
    }

    Ok(outputs)
}

fn extract_next_token_logits(session: &Session, outputs: Vec<Value>) -> anyhow::Result<Vec<f32>> {
    let logits_index = session
        .output_names()
        .iter()
        .position(|name| name == "logits")
        .or_else(|| {
            session
                .output_names()
                .iter()
                .position(|name| name.to_ascii_lowercase().contains("logits"))
        })
        .context("model did not expose a logits output")?;
    let logits = outputs
        .get(logits_index)
        .context("logits output index was out of range")?;
    let shape = logits.shape();
    let data = logits
        .to_vec_f32()
        .map_err(|e| anyhow::anyhow!("Failed to read logits tensor: {}", e))?;

    match shape {
        [vocab] if *vocab > 0 => Ok(data),
        [seq, vocab] if *seq > 0 && *vocab > 0 => {
            let vocab = *vocab as usize;
            let start = (*seq as usize - 1) * vocab;
            Ok(data[start..start + vocab].to_vec())
        }
        [batch, seq, vocab] if *batch > 0 && *seq > 0 && *vocab > 0 => {
            let vocab = *vocab as usize;
            let start = (*seq as usize - 1) * vocab;
            Ok(data[start..start + vocab].to_vec())
        }
        other => anyhow::bail!("unsupported logits tensor shape: {:?}", other),
    }
}

fn ensure_i64(info: &TensorInfo) -> anyhow::Result<()> {
    if info.dtype != DataType::Int64 {
        anyhow::bail!("input '{}' must be Int64, got {:?}", info.name, info.dtype);
    }
    Ok(())
}

fn empty_past_value(info: &TensorInfo) -> anyhow::Result<Value> {
    if info.dtype != DataType::Float32 {
        anyhow::bail!(
            "KV input '{}' must be Float32 for Phase 1, got {:?}",
            info.name,
            info.dtype
        );
    }
    if info.shape.len() < 3 {
        anyhow::bail!(
            "KV input '{}' has unsupported shape {:?}",
            info.name,
            info.shape
        );
    }
    let seq_axis = info.shape.len() - 2;
    let mut shape = Vec::with_capacity(info.shape.len());
    for (axis, &dim) in info.shape.iter().enumerate() {
        let value = if axis == 0 {
            1
        } else if axis == seq_axis {
            0
        } else if dim > 0 {
            dim
        } else {
            anyhow::bail!(
                "cannot infer static dimension {} for empty KV input '{}' shape {:?}",
                axis,
                info.name,
                info.shape
            );
        };
        shape.push(value);
    }
    Value::from_slice_f32(&[], &shape)
        .map_err(|e| anyhow::anyhow!("Failed to create empty KV input '{}': {}", info.name, e))
}

fn clone_value(value: &Value) -> anyhow::Result<Value> {
    match value.dtype() {
        DataType::Float32 => Value::from_slice_f32(&value.to_vec_f32()?, value.shape())
            .map_err(|e| anyhow::anyhow!("Failed to clone Float32 ORT value: {}", e)),
        DataType::Int64 => Value::from_slice_i64(&value.to_vec_i64()?, value.shape())
            .map_err(|e| anyhow::anyhow!("Failed to clone Int64 ORT value: {}", e)),
        dtype => anyhow::bail!("unsupported cached ORT value dtype: {:?}", dtype),
    }
}

fn is_kv_input(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains("past") && (lower.contains("key") || lower.contains("value"))
}

fn is_present_output(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains("present") && (lower.contains("key") || lower.contains("value"))
}

fn matching_past_input<'a>(present_name: &str, inputs: &'a [String]) -> Option<&'a String> {
    let present_suffix = kv_suffix(present_name)?;
    inputs
        .iter()
        .find(|input| kv_suffix(input).as_deref() == Some(present_suffix.as_str()))
}

fn kv_suffix(name: &str) -> Option<String> {
    let lower = name.to_ascii_lowercase();
    for prefix in [
        "past_key_values.",
        "present_key_values.",
        "past.",
        "present.",
    ] {
        if let Some(suffix) = lower.strip_prefix(prefix) {
            return Some(suffix.to_string());
        }
    }
    None
}

fn build_processor_chain(options: &GenerateOptions) -> ProcessorChain {
    let mut chain = ProcessorChain::new();

    if options.repetition_penalty != 1.0 {
        chain.add(Box::new(RepetitionPenaltyProcessor {
            penalty: options.repetition_penalty,
        }));
    }

    if !options.stop_sequences.is_empty() {
        chain.add(Box::new(StopSequenceProcessor::new(
            options.stop_sequences.clone(),
        )));
    }

    if options.temperature > 0.0 && options.temperature != 1.0 {
        chain.add(Box::new(TemperatureProcessor {
            temperature: options.temperature,
        }));
    }

    if options.top_k > 0 {
        chain.add(Box::new(TopKProcessor {
            top_k: options.top_k,
        }));
    }

    if options.top_p < 1.0 {
        chain.add(Box::new(TopPProcessor {
            top_p: options.top_p,
        }));
    }

    chain
}

fn select_next_token(
    logits: &mut [f32],
    context: &ProcessorContext,
    options: &GenerateOptions,
    chain: &ProcessorChain,
    rng_value: f32,
) -> TokenId {
    chain.process(logits, context);
    if options.greedy || options.temperature == 0.0 {
        sample_greedy(logits)
    } else {
        sample_categorical(logits, rng_value)
    }
}

fn finish_reason_after_token(
    token_id: TokenId,
    options: &GenerateOptions,
    chain: &ProcessorChain,
    context: &ProcessorContext,
) -> Option<FinishReason> {
    if options.stop_on_eos && options.eos_token_id == Some(token_id) {
        return Some(FinishReason::EosToken);
    }

    match chain.signal(context) {
        Some(ProcessorSignal::StopSequence { index }) => Some(FinishReason::StopSequence { index }),
        None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn processor_chain_uses_documented_order() {
        let options = GenerateOptions {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 10,
            repetition_penalty: 1.1,
            stop_sequences: vec![StopSequence::Tokens(vec![42])],
            ..Default::default()
        };
        let chain = build_processor_chain(&options);
        assert_eq!(
            chain.names(),
            vec![
                "repetition_penalty",
                "stop_sequence",
                "temperature",
                "top_k",
                "top_p"
            ]
        );
    }

    #[test]
    fn greedy_selection_uses_argmax_after_processors() {
        let options = GenerateOptions {
            greedy: true,
            top_k: 2,
            ..Default::default()
        };
        let chain = build_processor_chain(&options);
        let context = ProcessorContext::default();
        let mut logits = vec![0.0, 2.0, 4.0, 3.0];
        assert_eq!(
            select_next_token(&mut logits, &context, &options, &chain, 0.0),
            2
        );
    }

    #[test]
    fn sampled_selection_can_pick_non_argmax() {
        let options = GenerateOptions {
            greedy: false,
            ..Default::default()
        };
        let chain = build_processor_chain(&options);
        let context = ProcessorContext::default();
        let mut logits = vec![0.0, 0.0];
        assert_eq!(
            select_next_token(&mut logits, &context, &options, &chain, 0.75),
            1
        );
    }

    #[test]
    fn finish_reason_detects_eos_before_stop_sequence() {
        let options = GenerateOptions {
            eos_token_id: Some(7),
            stop_sequences: vec![StopSequence::Tokens(vec![7])],
            ..Default::default()
        };
        let chain = build_processor_chain(&options);
        let context = ProcessorContext {
            generated_tokens: vec![7],
            ..Default::default()
        };
        assert_eq!(
            finish_reason_after_token(7, &options, &chain, &context),
            Some(FinishReason::EosToken)
        );
    }

    #[test]
    fn finish_reason_detects_stop_sequence() {
        let options = GenerateOptions {
            stop_sequences: vec![StopSequence::Tokens(vec![2, 3])],
            ..Default::default()
        };
        let chain = build_processor_chain(&options);
        let context = ProcessorContext {
            generated_tokens: vec![1, 2, 3],
            ..Default::default()
        };
        assert_eq!(
            finish_reason_after_token(3, &options, &chain, &context),
            Some(FinishReason::StopSequence { index: 0 })
        );
    }

    #[test]
    fn tiny_fixture_generates_requested_tokens_end_to_end() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut engine = Engine::from_dir(&fixture, EngineConfig::default())?;
        let mut request = GenerateRequest::new("hello");
        request.options.max_new_tokens = 3;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;

        let result = engine.generate(request)?;

        assert_eq!(result.token_ids.len(), 3);
        assert_eq!(result.finish_reason, FinishReason::MaxTokens);
        Ok(())
    }
}

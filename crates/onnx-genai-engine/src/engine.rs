//! Main generation engine.

use crate::logits::{
    ProcessorChain, ProcessorContext, ProcessorSignal, RepetitionPenaltyProcessor, StopSequence,
    StopSequenceProcessor, TemperatureProcessor, TokenId, TopKProcessor, TopPProcessor,
};
use crate::sampling::{sample_categorical, sample_greedy};
use onnx_genai_kv::{PagedKvCache, SequenceId};
use onnx_genai_metadata::InferenceMetadata;
use onnx_genai_scheduler::{Priority, Scheduler, SchedulerConfig};
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
    /// Raw prompt text. Tokenization is wired when the tokenizer lands.
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
    /// Temperature applied before sampling. Must be positive for sampled generation.
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
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            anyhow::bail!("temperature must be finite and greater than zero");
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
    /// Detokenized generated text. Empty until tokenizer wiring lands for token-id-only paths.
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
    // ORT session (added when wiring up C API)
    // session: onnx_genai_ort::Session,
    // Tokenizer (added when wiring up HF tokenizers)
    // tokenizer: tokenizers::Tokenizer,
}

impl Engine {
    /// Load a model from a directory.
    pub fn from_dir(model_dir: &Path, config: EngineConfig) -> anyhow::Result<Self> {
        // Load metadata
        let metadata_path = model_dir.join("inference_metadata.yaml");
        let metadata = if metadata_path.exists() {
            onnx_genai_metadata::load_metadata(&metadata_path)
                .map_err(|e| anyhow::anyhow!("Failed to load metadata: {}", e))?
        } else {
            let json_path = model_dir.join("inference_metadata.json");
            if json_path.exists() {
                onnx_genai_metadata::load_metadata(&json_path)
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

        Ok(Self {
            metadata,
            kv_cache,
            scheduler,
        })
    }

    /// Generate text for a request.
    ///
    /// Phase 1 public API and sampling/stop handling are wired here. Tokenization,
    /// detokenization, and ORT forward execution are the only intentionally stubbed calls.
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
        let prompt_tokens = self.tokenize_prompt(&request.prompt)?;
        let seq_id = self.create_session();
        self.scheduler.add_request(
            seq_id,
            prompt_tokens.len(),
            request.options.max_new_tokens,
            Priority::Normal,
        );

        let chain = build_processor_chain(&request.options);
        let mut generated_tokens = Vec::new();
        let mut generated_text = String::new();

        for step in 0..request.options.max_new_tokens {
            let mut context = ProcessorContext {
                prompt_tokens: prompt_tokens.clone(),
                generated_tokens: generated_tokens.clone(),
                generated_text: generated_text.clone(),
                step,
            };

            let mut logits = self.next_token_logits(seq_id, &prompt_tokens, &generated_tokens)?;
            let token_id = select_next_token(&mut logits, &context, &request.options, &chain, 0.0);
            generated_tokens.push(token_id);
            self.scheduler.advance(seq_id);

            let token_text = self.detokenize_token(token_id)?;
            generated_text.push_str(&token_text);
            context.generated_tokens = generated_tokens.clone();
            context.generated_text = generated_text.clone();

            let finish_reason =
                finish_reason_after_token(token_id, &request.options, &chain, &context);
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
                    text: generated_text,
                    token_ids: generated_tokens,
                    finish_reason,
                });
            }
        }

        self.scheduler.complete(seq_id);
        Ok(GenerateResult {
            text: generated_text,
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
            GeneratePrompt::Text(_text) => {
                todo!("wire tokenizer from onnx-genai-ort/model loader in the next batch")
            }
        }
    }

    fn detokenize_token(&self, _token_id: TokenId) -> anyhow::Result<String> {
        todo!("wire tokenizer detokenization from onnx-genai-ort/model loader in the next batch")
    }

    fn next_token_logits(
        &mut self,
        _seq_id: SequenceId,
        _prompt_tokens: &[TokenId],
        _generated_tokens: &[TokenId],
    ) -> anyhow::Result<Vec<f32>> {
        todo!("wire ORT session forward pass from onnx-genai-ort in the next batch")
    }
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

    if options.temperature != 1.0 {
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
    if options.greedy {
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
}

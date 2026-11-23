//! Shared execution and cancellation for non-interactive CLI commands.
use super::profile;
use anyhow::Context as _;
use onnx_genai::engine::{EngineDecodeBackend, PipelineGenerateRequest, is_missing_required_input};
use onnx_genai::metadata::GenerationDefaults;
use onnx_genai::ort::{SessionOptions, Tokenizer};
use onnx_genai::{
    Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest, GenerateResult,
    GenerateTokenCallback,
};
use onnx_genai_server::multimodal::{self, MultimodalInput, MultimodalSpecs};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Once;
use std::sync::atomic::{AtomicBool, Ordering};

pub(super) const EXIT_INTERRUPTED: i32 = 130;
pub(super) static GENERATING: AtomicBool = AtomicBool::new(false);
pub(super) static INTERRUPT_REQUESTED: AtomicBool = AtomicBool::new(false);
static CTRLC_HANDLER: Once = Once::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum InputMode {
    Plain,
    Tty,
}

pub(super) fn input_mode(stdin_is_terminal: bool, stdout_is_terminal: bool) -> InputMode {
    if stdin_is_terminal && stdout_is_terminal {
        InputMode::Tty
    } else {
        InputMode::Plain
    }
}

pub(super) fn initial_show_stats(mode: InputMode, no_stats: bool) -> bool {
    matches!(mode, InputMode::Tty) && !no_stats
}

#[derive(Debug)]
pub(super) struct Interrupted;

impl fmt::Display for Interrupted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("generation interrupted")
    }
}

impl std::error::Error for Interrupted {}

/// Returns true when `error` was produced by a Ctrl-C interrupt (i.e. carries an
/// [`Interrupted`] marker), as opposed to a real generation failure.
pub(super) fn is_interrupt_error(error: &anyhow::Error) -> bool {
    error.downcast_ref::<Interrupted>().is_some()
}

pub(super) fn install_ctrlc_handler() {
    CTRLC_HANDLER.call_once(|| {
        if let Err(error) = ctrlc::set_handler(|| {
            if GENERATING.load(Ordering::SeqCst)
                && !INTERRUPT_REQUESTED.swap(true, Ordering::SeqCst)
            {
                return;
            }
            let _ = crate::flush_deferred_tracing();
            std::process::exit(EXIT_INTERRUPTED);
        }) {
            eprintln!("warning: could not install Ctrl-C handler: {error}");
        }
    });
}

/// One turn's input: the rendered prompt plus any attachments staged for it.
#[derive(Debug, Default)]
pub(super) struct TurnInput {
    pub(super) prompt: String,
    pub(super) images: Vec<PathBuf>,
    pub(super) audio: Vec<PathBuf>,
    pub(super) options: GenerateOptions,
    pub(super) prompt_tokens: Option<usize>,
    pub(super) context_limit: Option<usize>,
}

/// The loaded model, and the contracts needed to feed it.
///
/// One backend, because there is one runtime. What varies between packages is
/// what they *declare*: a package whose metadata declares image or audio input
/// carries [`Self::multimodal`], and a turn with an attachment is bound
/// against those contracts. Nothing here asks which executor runs the model.
pub(super) struct Backend {
    engine: Engine,
    tokenizer: Option<Tokenizer>,
    multimodal: Option<MultimodalSpecs>,
    generation_defaults: Option<GenerationDefaults>,
}

impl Backend {
    /// Load `model_dir`, preferring its declared pipeline when it has one.
    pub(super) fn load(model_dir: &Path, config: EngineConfig) -> anyhow::Result<Self> {
        Self::load_with_options(model_dir, config, SessionOptions::default())
    }

    pub(super) fn load_with_options(
        model_dir: &Path,
        config: EngineConfig,
        session_options: SessionOptions,
    ) -> anyhow::Result<Self> {
        match multimodal::load(model_dir)? {
            Some(setup) => {
                let tokenizer = Tokenizer::from_file(&setup.tokenizer_path).map_err(|error| {
                    anyhow::anyhow!(
                        "What: the pipeline's prompt tokenizer could not be loaded from {}. \
                         Why: {error}. \
                         How: verify the package ships a valid tokenizer.json.",
                        setup.tokenizer_path.display()
                    )
                })?;
                let engine =
                    Engine::from_dir_with_session_options(model_dir, config, session_options)?;
                Ok(Self {
                    engine,
                    tokenizer: Some(tokenizer),
                    multimodal: Some(setup.multimodal),
                    generation_defaults: setup.generation_defaults,
                })
            }
            None => Ok(Self {
                engine: Engine::from_dir_with_session_options(model_dir, config, session_options)?,
                tokenizer: None,
                multimodal: None,
                generation_defaults: None,
            }),
        }
    }

    /// Declared non-text input contracts, when the package declares any.
    pub(super) fn multimodal(&self) -> Option<&MultimodalSpecs> {
        self.multimodal.as_ref()
    }

    /// Run one turn, streaming tokens through `callback`.
    /// Clear the pipeline reuse counters so a profile covers only the next turn.
    pub(super) fn reset_reuse_stats(&self) {}

    /// What a multimodal pipeline avoided recomputing, or `None` for a single
    /// decoder graph, which has no encoder or attachments to reuse.
    pub(super) fn multimodal_reuse(&self) -> Option<profile::MultimodalReuse> {
        None
    }

    /// What the KV page pool holds right now, when the backend pages its KV.
    pub(super) fn page_stats(&self) -> Option<onnx_genai::kv::PageStats> {
        self.engine.pages_kv().then(|| self.engine.page_stats())
    }

    /// Concrete decoder backend selected for the loaded model.
    pub(super) fn decode_backend(&self) -> EngineDecodeBackend {
        self.engine.decode_backend()
    }

    /// Execution-provider placement reported by the loaded model, not by the
    /// requested settings.
    pub(super) fn execution_provider_status(&self) -> String {
        self.engine.execution_provider_status()
    }

    /// KV-cache accounting from the engine's resource governor.
    ///
    /// Only a single-model engine runs a governor; a pipeline reports nothing
    /// rather than a zero that would read as "no KV cache".
    pub(super) fn kv_usage(&self) -> Option<profile::MemoryUsage> {
        {
            let engine = &self.engine;
            if !engine.pages_kv() {
                return Some(profile::MemoryUsage {
                    memory_strategy_plan: Some(engine.memory_strategy_plan().clone()),
                    ..profile::MemoryUsage::default()
                });
            }
            {
                let snapshot = engine.resource_snapshot();
                let budget = snapshot.derived_budget;
                let breakdown = snapshot.breakdown;
                #[cfg(feature = "native-backend")]
                let weight_placement =
                    engine
                        .weight_placement_report()
                        .map(|report| profile::WeightPlacementMemory {
                            coordinated_weight_budget_bytes: report.coordinated_weight_budget_bytes,
                            effective_budget_bytes: report.effective_budget_bytes,
                            device_bytes: report.device_bytes,
                            host_bytes: report.host_bytes,
                            explanation: report.explanation.clone(),
                        });
                #[cfg(not(feature = "native-backend"))]
                let weight_placement = None;
                Some(profile::MemoryUsage {
                    kv_budget_bytes: Some(budget.kv_bytes),
                    kv_max_tokens: Some(budget.max_total_tokens),
                    host_ram_used_bytes: Some(snapshot.host_ram.used),
                    device_used_bytes: Some(snapshot.vram.used),
                    device_limit_bytes: snapshot.resolved_limits.vram_bytes,
                    device_oversubscribed_bytes: Some(engine.device_oversubscribed_bytes()),
                    peak_resident_bytes: None,
                    composition: Some(profile::DeviceComposition {
                        model_weights_bytes: breakdown.model_weights_bytes,
                        activations_bytes: breakdown.activations_bytes,
                        runtime_overhead_bytes: breakdown.ort_overhead_bytes,
                        kv_bytes: budget.kv_bytes,
                        kv_pages: budget.total_pages,
                        kv_page_bytes: budget.kv_bytes.checked_div(budget.total_pages).unwrap_or(0),
                    }),
                    activation_plan: engine.activation_memory_plan_stats().map(|stats| {
                        profile::ActivationPlanMemory {
                            complete: stats.complete,
                            peak_bytes: stats.peak_bytes,
                            naive_bytes: stats.naive_bytes,
                            savings_ratio: stats.savings_ratio,
                            unknown_sizes: stats.unknown_sizes,
                        }
                    }),
                    weight_placement,
                    memory_strategy_plan: Some(engine.memory_strategy_plan().clone()),
                    vmm_arena: engine.vmm_arena_stats().map(|stats| profile::VmmArena {
                        commits: stats.commits,
                        releases: stats.releases,
                        committed_bytes: stats.committed_bytes,
                        reserved_bytes: stats.reserved_bytes,
                        peak_committed_bytes: stats.peak_committed_bytes,
                        allocations: stats.allocations,
                        ref_underflows: stats.ref_underflows,
                        byte_underflows: stats.byte_underflows,
                        unaccounted_committed_bytes: stats.unaccounted_committed_bytes,
                    }),
                })
            }
        }
    }

    /// Number of tokens the prompt occupies, when the backend can tell.
    pub(super) fn prompt_tokens(&self, prompt: &str) -> Option<usize> {
        match self.tokenizer.as_ref() {
            Some(tokenizer) => tokenizer.encode(prompt).ok().map(|ids| ids.len()),
            None => self.engine.tokenize(prompt).ok().map(|ids| ids.len()),
        }
    }

    /// Token id for an exact one-token marker string, when the loaded tokenizer
    /// represents it as one token.
    pub(super) fn single_token_id(&self, token: &str) -> Option<u32> {
        let ids = match self.tokenizer.as_ref() {
            Some(tokenizer) => tokenizer
                .token_id(token)
                .map(|id| vec![id])
                .or_else(|| tokenizer.encode(token).ok())?,
            None => self.engine.tokenize(token).ok()?,
        };
        match ids.as_slice() {
            [id] => Some(*id),
            _ => None,
        }
    }

    pub(super) fn bind_reasoning_marker_tokens(
        &self,
        reasoning: &mut Option<super::output::ReasoningConfig>,
    ) {
        if let Some(config) = reasoning {
            let start_id = self.single_token_id(&config.markers.start);
            let end_id = self.single_token_id(&config.markers.end);
            config.set_marker_token_ids(start_id, end_id);
        }
    }

    pub(super) fn effective_max_context(&self, options: &GenerateOptions) -> Option<usize> {
        self.engine.effective_max_context(options)
    }

    /// The package's declared generation defaults, when it declared any.
    ///
    /// A package states its sampling regime (a reasoning model shipping
    /// `do_sample: true`, or a legacy `search` block imported into
    /// `generation.defaults`) as part of what the model *is*. Discarding it
    /// silently reinterprets every such package as greedy, which is not a
    /// neutral default: a model tuned to sample can degenerate into repetition
    /// under argmax. Explicit CLI flags still win — this only supplies the
    /// values the caller left unstated.
    pub(super) fn generation_defaults(&self) -> Option<&GenerationDefaults> {
        self.generation_defaults.as_ref().or_else(|| {
            self.engine
                .metadata()
                .generation
                .as_ref()
                .and_then(|generation| generation.defaults.as_ref())
        })
    }

    pub(super) fn generate(
        &mut self,
        turn: TurnInput,
        callback: &mut GenerateTokenCallback<'_>,
    ) -> anyhow::Result<GenerateResult> {
        multimodal::admit_attachments(
            self.multimodal(),
            "the loaded model",
            turn.images.len(),
            turn.audio.len(),
        )?;
        let (Some(tokenizer), Some(multimodal)) =
            (self.tokenizer.as_ref(), self.multimodal.as_ref())
        else {
            let request = GenerateRequest {
                prompt: GeneratePrompt::Text(turn.prompt),
                options: turn.options,
            };
            return self.engine.generate_with_callback(request, Some(callback));
        };
        let attachments = turn.images.len() + turn.audio.len();
        let required = multimodal.sole_modality();
        let request = build_pipeline_request(tokenizer, multimodal, turn)?;
        self.engine
            .generate_with_pipeline_callbacks(request, None, Some(callback))
            .map_err(|error| match required {
                // A multimodal package can require its non-text input; say so
                // rather than leaving a bare "missing input".
                Some(modality) if attachments == 0 && is_missing_required_input(&error) => error
                    .context(format!(
                        "the turn carried no attachment, but this model declares {modality} input. \
                         How: provide `--{modality} <path>` on the command line."
                    )),
                _ => error,
            })
    }
}

pub(super) fn apply_context_sized_max_new_tokens(
    options: &mut GenerateOptions,
    max_new_tokens_was_explicit: bool,
    prompt_tokens: usize,
    effective_max_context: Option<usize>,
) -> bool {
    if max_new_tokens_was_explicit {
        return false;
    }
    match effective_max_context {
        Some(limit) => {
            // Leave the context-window check, not max_new_tokens, as the natural
            // stop so the engine reports FinishReason::Length at the boundary.
            options.max_new_tokens = limit.saturating_sub(prompt_tokens).saturating_add(1);
            false
        }
        None => {
            options.max_new_tokens = super::CLI_FALLBACK_MAX_NEW_TOKENS;
            true
        }
    }
}

pub(super) fn warn_missing_context_limit(max_new_tokens: usize) {
    eprintln!(
        "warning: model context length could not be inferred from inference metadata or the decode path; using finite fallback --max-new-tokens {max_new_tokens}. Configure --max-context, or declare model.max_sequence_length in inference metadata, to generate until the context window is full without risking an ORT out-of-bounds decode."
    );
}

pub(super) fn context_exhaustion_error(prompt_tokens: usize, limit: usize) -> anyhow::Error {
    anyhow::anyhow!(
        "What: the prompt already fills the context window ({prompt_tokens}/{limit} tokens). \
         Why: there is no room to generate even one token, so returning an empty answer would be misleading. \
         How: shorten the prompt or use a model/package with a larger context window."
    )
}

pub(super) fn context_window_is_full(
    prompt_tokens: usize,
    effective_max_context: Option<usize>,
) -> Option<usize> {
    effective_max_context.filter(|&limit| prompt_tokens >= limit)
}

/// Turn a prompt plus its attachments into a pipeline generation request.
/// Audio replaces the prompt entirely: the transcription decoder is seeded with
/// the model's own transcription token sequence because the spoken audio, not
/// the typed text, carries the content. Images keep the prompt and expand each
/// placeholder token into the declared image-token run.
pub(super) fn build_pipeline_request(
    tokenizer: &Tokenizer,
    multimodal: &MultimodalSpecs,
    turn: TurnInput,
) -> anyhow::Result<PipelineGenerateRequest> {
    let TurnInput {
        prompt,
        images,
        audio,
        options,
        prompt_tokens: _,
        context_limit: _,
    } = turn;

    if let Some(path) = audio.first() {
        let spec = multimodal
            .audio
            .as_ref()
            .expect("audio support checked before building the request");
        let input = MultimodalInput::from_wav(spec, &read_attachment(path, "audio")?)
            .with_context(|| {
                format!(
                    "What: the audio file {} could not be preprocessed. \
                     Why: it is not usable as this model's declared audio input. \
                     How: provide a PCM16 WAV file.",
                    path.display()
                )
            })?;
        // Audio replaces the prompt: the transcription decoder is seeded with
        // the model's own token sequence because the clip, not the typed text,
        // carries the content.
        let token_ids = multimodal::audio_decoder_prompt(tokenizer, None)?;
        return input.bind(PipelineGenerateRequest::new(GenerateRequest {
            prompt: GeneratePrompt::TokenIds(token_ids),
            options,
        }));
    }

    let mut token_ids = tokenizer.encode(&prompt).map_err(|error| {
        anyhow::anyhow!(
            "What: the prompt could not be tokenized. \
             Why: {error}. \
             How: verify the package's tokenizer.json matches its decoder."
        )
    })?;
    let input = match multimodal.vision.as_ref().filter(|_| !images.is_empty()) {
        None => None,
        Some(spec) => {
            let mut encoded = Vec::with_capacity(images.len());
            for path in &images {
                encoded.push(read_attachment(path, "image")?);
            }
            Some(MultimodalInput::from_images(
                spec,
                &encoded,
                &mut token_ids,
                multimodal::MAX_EXPANDED_PROMPT_TOKENS,
            )?)
        }
    };

    let request = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(token_ids),
        options,
    });
    match input {
        Some(input) => input.bind(request),
        None => Ok(request),
    }
}

/// Read an attachment file, naming it and its modality on failure.
pub(super) fn read_attachment(path: &Path, kind: &str) -> anyhow::Result<Vec<u8>> {
    std::fs::read(path).map_err(|error| {
        anyhow::anyhow!(
            "What: the {kind} file {} could not be read. \
             Why: {error}. \
             How: check the path and that the file is readable.",
            path.display()
        )
    })
}

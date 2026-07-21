//! Multi-model pipeline orchestrator.

use crate::decode::{
    DecodeState, clone_value, extract_next_token_logits, is_present_output, is_token_input_name,
    run_decode_step_with_extra,
};
use crate::decode_loop::{DecodeLoopBackend, DecodeLoopState, run_decode_loop};
use crate::engine::{
    Engine, EngineConfig, model_requires_native_backend, requested_decode_backend,
};
use crate::kv_bridge::infer_kv_model_info;
use crate::logits::TokenId;
use crate::processors::build_processor_chain;
use crate::{
    EngineDecodeBackend, GeneratePrompt, GenerateRequest, GenerateResult, GenerateTokenCallback,
};
use anyhow::Context;
use onnx_genai_metadata::{
    DataflowEdge, PhaseRunOn, PipelineSpec, PipelineStrategy, PipelineStrategyKind,
    PipelineVisionConfig, SchedulerSpec,
};
use onnx_genai_ort::{
    DataType, PipelineModelDirectory, PipelineModels, Session, SessionOptions, Tokenizer, Value,
};
use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex};
use std::path::Path;

/// Named tensors supplied to or produced by pipeline components.
///
/// Keys are fully-qualified endpoints of the form `component.input_name` or
/// `component.output_name`.
pub type PipelineTensors = HashMap<String, Value>;

/// The result of a post-decode (text-to-speech-shaped) pipeline run: the
/// autoregressive decoder's generated code tokens plus the final tensor pool
/// produced by the post-decode single-pass stages (e.g. a vocoder waveform).
///
/// Returned by [`PipelineEngine::synthesize`]. The `generation` field carries
/// the code token ids (as [`generate`](PipelineEngine::generate) would return),
/// while `tensors` holds every stage output keyed by `component.output` —
/// including the synthetic `{decoder}.output_ids` codes tensor and the vocoder's
/// waveform (e.g. `vocoder.audio`).
pub struct PipelineSynthesis {
    /// The AR decoder's generated code tokens and finish metadata.
    pub generation: GenerateResult,
    /// The shared tensor pool after the post-decode stages ran.
    pub tensors: PipelineTensors,
}

/// Per-request overrides for an iterative (diffusion) pipeline's loop
/// parameters. This enables ComfyUI-style *live* editing — re-driving the same
/// already-loaded models with different dynamics, with no re-export or reload.
///
/// The seed, prompt and negative prompt are already live: they are supplied as
/// per-request inputs (`denoiser.sample`, `text_encoder.input_ids`, and any
/// `*.uncond` conditioning), so only the loop *parameters* need overrides here.
#[derive(Debug, Clone, Default)]
pub struct IterativeOverrides {
    /// Override the number of denoise steps. Rebuilds the scheduler for the new
    /// step count; rejected when the pipeline declares an explicit per-step
    /// timestep schedule (which is tied to the original step count).
    pub num_steps: Option<usize>,
    /// Override the classifier-free-guidance scale (ComfyUI `cfg`). `1.0`
    /// disables guidance.
    pub guidance_scale: Option<f32>,
    /// Override the first step index of a partial (img2img) denoise loop.
    pub start_step: Option<usize>,
}

/// A pipeline generation request.
pub struct PipelineGenerateRequest {
    pub request: GenerateRequest,
    /// External tensors keyed by `component.input_name`.
    pub inputs: PipelineTensors,
    /// Number of image tiles represented by the external vision tensor.
    ///
    /// This is known only after preprocessing and must be supplied before
    /// decoder KV allocation for encoder-free multimodal pipelines.
    pub num_image_tiles: Option<usize>,
    /// Live overrides for an iterative pipeline's loop parameters.
    pub iterative_overrides: IterativeOverrides,
}

impl PipelineGenerateRequest {
    pub fn new(request: GenerateRequest) -> Self {
        Self {
            request,
            inputs: HashMap::new(),
            num_image_tiles: None,
            iterative_overrides: IterativeOverrides::default(),
        }
    }

    pub fn with_input(mut self, endpoint: impl Into<String>, value: Value) -> Self {
        self.inputs.insert(endpoint.into(), value);
        self
    }

    pub fn with_image_tile_count(mut self, num_image_tiles: usize) -> Self {
        self.num_image_tiles = Some(num_image_tiles);
        self
    }

    /// Attach live overrides for an iterative pipeline's loop parameters
    /// (steps / guidance scale / start step).
    pub fn with_iterative_overrides(mut self, overrides: IterativeOverrides) -> Self {
        self.iterative_overrides = overrides;
        self
    }
}

impl From<GenerateRequest> for PipelineGenerateRequest {
    fn from(request: GenerateRequest) -> Self {
        Self::new(request)
    }
}

/// Engine for metadata-declared multi-model pipelines.
pub struct PipelineEngine {
    models: PipelineModels,
    plan: PipelinePlan,
    /// Autoregressive decode state; `None` for non-autoregressive pipelines
    /// (single-pass, iterative/diffusion) which produce tensors, not tokens.
    decoder_state: Option<DecodeState>,
    tokenizer_component: String,
}

impl Engine {
    /// Load a metadata-declared pipeline directory.
    ///
    /// The returned [`PipelineEngine`] keeps the existing single-model `Engine`
    /// API stable while exposing a separate end-to-end pipeline path.
    pub fn from_pipeline_dir(
        pipeline_dir: &Path,
        config: EngineConfig,
    ) -> anyhow::Result<PipelineEngine> {
        PipelineEngine::from_dir_with_config(pipeline_dir, config)
    }

    /// Load a pipeline directory with a custom [`SchedulerRegistry`] so users
    /// can plug in their own [`Scheduler`] implementations.
    pub fn from_pipeline_dir_with_schedulers(
        pipeline_dir: &Path,
        config: EngineConfig,
        schedulers: &SchedulerRegistry,
    ) -> anyhow::Result<PipelineEngine> {
        PipelineEngine::from_dir_with_schedulers(pipeline_dir, config, schedulers)
    }
}

impl PipelineEngine {
    /// Load all pipeline sessions with default CPU ORT options.
    pub fn from_dir(pipeline_dir: &Path) -> anyhow::Result<Self> {
        Self::from_dir_with_config(pipeline_dir, EngineConfig::default())
    }

    pub fn from_dir_with_config(pipeline_dir: &Path, config: EngineConfig) -> anyhow::Result<Self> {
        Self::from_dir_with_schedulers(pipeline_dir, config, &SchedulerRegistry::builtin())
    }

    /// Load a pipeline with a **custom [`SchedulerRegistry`]**, so a user can
    /// plug in their own [`Scheduler`] implementations (referenced by
    /// `scheduler_config.kind` in the pipeline metadata) alongside the built-in
    /// `ddim` / `masked_diffusion`.
    pub fn from_dir_with_schedulers(
        pipeline_dir: &Path,
        config: EngineConfig,
        schedulers: &SchedulerRegistry,
    ) -> anyhow::Result<Self> {
        let decode_backend = requested_decode_backend(config.decode_backend)?;
        if decode_backend == EngineDecodeBackend::Native {
            anyhow::bail!(
                "native backend not supported for pipeline models; \
                 set decode_backend = EngineDecodeBackend::Ort (or ONNX_GENAI_BACKEND=ort)"
            );
        }
        if decode_backend == EngineDecodeBackend::Auto {
            let directory = PipelineModelDirectory::load(pipeline_dir)
                .map_err(|e| anyhow::anyhow!("Failed to resolve pipeline models: {}", e))?;
            for (component, model_path) in &directory.model_paths {
                if model_requires_native_backend(model_path)? {
                    anyhow::bail!(
                        "native backend not supported for pipeline models: component '{component}' requires the native backend"
                    );
                }
            }
        }
        let models = PipelineModels::load_with_options(pipeline_dir, SessionOptions::default())
            .map_err(|e| anyhow::anyhow!("Failed to load pipeline models: {}", e))?;
        let plan = PipelinePlan::from_spec(&models.directory.spec, schedulers)?;
        // Only autoregressive pipelines drive a token-by-token decode loop and
        // therefore need a `DecodeState` + KV model info. Single-pass and
        // iterative (diffusion) pipelines run tensors through `run_pipeline`.
        let (decoder_state, tokenizer_component) = match &plan {
            PipelinePlan::Autoregressive(ar) => {
                let decoder = models.session(&ar.decoder).with_context(|| {
                    format!("pipeline decoder '{}' was not loaded", ar.decoder)
                })?;
                let _kv_model =
                    infer_kv_model_info(decoder, config.page_size, config.kv_cache_dtype)?;
                (Some(DecodeState::new(decoder)?), ar.decoder.clone())
            }
            // A nested-AR (multi-decoder TTS) pipeline drives its own outer/inner
            // decode loops with per-loop `DecodeState`s built inside the driver,
            // so no shared decode state is created here. The tokenizer component
            // is the outer decoder (talker).
            PipelinePlan::NestedAutoregressive(nested) => (None, nested.outer.clone()),
            PipelinePlan::SinglePass(sp) => (None, sp.model.clone()),
            PipelinePlan::Iterative(it) => (None, it.denoiser.clone()),
            // A pure composite produces tensors (run_pipeline), not text; it has
            // no autoregressive decode state. Use the last stage's model as the
            // nominal tokenizer component (unused unless a tokenizer is queried).
            PipelinePlan::Composite(c) => (
                None,
                c.stages
                    .last()
                    .map(|stage| match &stage.kind {
                        CompositeStageKind::SinglePass { model } => model.clone(),
                    })
                    .unwrap_or_default(),
            ),
        };
        Ok(Self {
            models,
            plan,
            decoder_state,
            tokenizer_component,
        })
    }

    /// Generate text from a pipeline with no extra non-text tensors.
    pub fn generate(&mut self, request: GenerateRequest) -> anyhow::Result<GenerateResult> {
        self.generate_with_pipeline_request(request.into())
    }

    /// Generate text while supplying external component inputs, such as
    /// `vision_encoder.pixel_values` for a VLM encoder.
    pub fn generate_with_pipeline_request(
        &mut self,
        pipeline_request: PipelineGenerateRequest,
    ) -> anyhow::Result<GenerateResult> {
        self.generate_with_callback(pipeline_request, None)
    }

    /// Generate text and optionally stream tokens.
    pub fn generate_with_callback(
        &mut self,
        pipeline_request: PipelineGenerateRequest,
        callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        // A nested-AR (multi-decoder TTS) pipeline drives its own outer/inner
        // loops; `generate` returns the flattened per-frame code tokens (use
        // `synthesize` to also run the post-decode vocoder into a waveform).
        if matches!(self.plan, PipelinePlan::NestedAutoregressive(_)) {
            return self
                .run_nested_autoregressive(pipeline_request)
                .map(|(result, _pool)| result);
        }
        self.run_autoregressive(pipeline_request, callback)
            .map(|(result, _pool)| result)
    }

    /// Run a **text-to-speech**-shaped pipeline: prompt-phase encoders, then the
    /// AR decode loop (which emits audio *code* tokens), then the post-decode
    /// `final_only` single-pass stages (a vocoder) that turn the collected codes
    /// into a waveform. Returns both the generated codes ([`GenerateResult`]) and
    /// the final tensor pool ([`PipelineTensors`], keyed by `component.output`),
    /// which holds the vocoder waveform (e.g. `vocoder.audio`).
    ///
    /// This is the post-decode-stage counterpart to [`generate`](Self::generate)
    /// (codes only) and [`run_pipeline`](Self::run_pipeline) (no AR loop). The AR
    /// decoder's generated code sequence is published into the shared pool as the
    /// synthetic tensor `{decoder}.output_ids` of shape `[1, num_generated]`
    /// (int64), so a post-decode stage consumes it via a dataflow edge such as
    /// `decoder.output_ids -> vocoder.codes`.
    pub fn synthesize(
        &mut self,
        pipeline_request: PipelineGenerateRequest,
    ) -> anyhow::Result<PipelineSynthesis> {
        // A nested-AR (multi-decoder TTS) pipeline publishes its assembled codes
        // as `{outer}.output_codes` inside its own driver; run the post-decode
        // vocoder over the shared pool separately.
        if matches!(self.plan, PipelinePlan::NestedAutoregressive(_)) {
            return self.synthesize_nested(pipeline_request);
        }
        let (generation, mut tensors) = self.run_autoregressive(pipeline_request, None)?;
        let ar = self.plan.autoregressive_plan()?.clone();

        // Publish the AR decoder's generated code sequence into the shared pool
        // as `{decoder}.output_ids` [1, num_generated] (int64) so a post-decode
        // single-pass stage can consume it via a dataflow edge.
        let codes: Vec<i64> = generation.token_ids.iter().map(|&t| i64::from(t)).collect();
        let codes_endpoint = format!("{}.output_ids", ar.decoder);
        let codes_value =
            Value::from_slice_i64(&codes, &[1, codes.len() as i64]).with_context(|| {
                format!("failed to build generated-codes tensor '{codes_endpoint}'")
            })?;
        tensors.insert(codes_endpoint, codes_value);

        // Run the post-decode `final_only` stages once, in declared order, over
        // the shared pool (codes + prompt-phase tensors), so the vocoder reads
        // the routed codes and writes its waveform back into the pool.
        self.run_prompt_phase_components(
            &ar.post_decode_components,
            &mut tensors,
            "postlogue",
            None,
        )?;
        Ok(PipelineSynthesis {
            generation,
            tensors,
        })
    }

    /// Core autoregressive execution shared by [`generate_with_callback`] and
    /// [`synthesize`]: run the prompt-phase components, drive the decode loop,
    /// and return the generated tokens alongside the shared tensor pool (external
    /// inputs + prompt-phase outputs) so a caller can run post-decode stages.
    fn run_autoregressive(
        &mut self,
        pipeline_request: PipelineGenerateRequest,
        callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<(GenerateResult, PipelineTensors)> {
        // Guard first: a non-autoregressive pipeline (single-pass / iterative
        // diffusion) has no token decode loop, so surface the actionable error
        // before touching the tokenizer or options.
        let ar = self
            .plan
            .autoregressive_plan()
            .context(
                "generate() requires an autoregressive pipeline; use run_pipeline() for \
                 single-pass or iterative (diffusion) pipelines",
            )?
            .clone();

        let mut options = pipeline_request.request.options.clone();
        options.validate()?;
        if options.eos_token_id.is_none() {
            options.eos_token_id = self.tokenizer()?.eos_token_id();
        }
        let prompt_tokens = tokenize_with(self.tokenizer()?, &pipeline_request.request.prompt)?;
        if prompt_tokens.is_empty() {
            anyhow::bail!("prompt must contain at least one token");
        }
        if pipeline_request.num_image_tiles == Some(0) {
            anyhow::bail!("image tile count must be greater than zero");
        }
        // TODO(#14): Pipeline metadata must declare the image placeholder token
        // and tokens-per-tile contract. Expand that placeholder here using
        // `num_image_tiles` before DecodeState/KV allocation. The server vision
        // seam should pass ImageTensor::num_tiles via with_image_tile_count().

        let prompt_tokens = expand_image_placeholders_count_based(
            prompt_tokens,
            pipeline_request.num_image_tiles,
            self.models.directory.spec.vision.as_ref(),
        )?;

        let mut tensors = pipeline_request.inputs;
        // Seed the prompt token ids into the shared pool so a prompt-phase
        // fusion component (Gemma4-style `embedding`) can consume `input_ids`.
        self.seed_prompt_token_inputs(&ar.prompt_components, &prompt_tokens, &mut tensors)?;
        self.run_prompt_phase_components(&ar.prompt_components, &mut tensors, "prologue", None)?;
        // A decoder whose prompt input is `inputs_embeds` (not `input_ids`) needs
        // the fusion component re-run each step to embed the running token; the
        // routed prompt embeddings are otherwise stale after prefill.
        let embeds_binding = self.embeds_step_binding(&ar.decoder, &tensors)?;
        let decoder_extras = self.decoder_extra_inputs(
            &ar.decoder,
            &tensors,
            embeds_binding.as_ref().map(|b| b.decoder_input.as_str()),
        )?;

        let chain = build_processor_chain(&options, Some(self.tokenizer()?))?;
        self.decoder_state = Some({
            let decoder = self
                .models
                .session(&ar.decoder)
                .with_context(|| format!("pipeline decoder '{}' was not loaded", ar.decoder))?;
            DecodeState::new(decoder)?
        });

        let decoder = self
            .models
            .session(&ar.decoder)
            .with_context(|| format!("pipeline decoder '{}' was not loaded", ar.decoder))?;
        let embed_session = match &embeds_binding {
            Some(binding) => Some(
                self.models
                    .session(&binding.component)
                    .with_context(|| {
                        format!(
                            "pipeline embedding component '{}' was not loaded",
                            binding.component
                        )
                    })?,
            ),
            None => None,
        };
        let tokenizer = self
            .models
            .tokenizer_for(&self.tokenizer_component)
            .with_context(|| {
                format!("no tokenizer available for '{}'", self.tokenizer_component)
            })?;
        let mut backend = PipelineDecodeLoopBackend {
            decoder,
            decoder_state: self
                .decoder_state
                .as_mut()
                .expect("autoregressive pipeline has decode state"),
            decoder_extras: &decoder_extras,
            embed_session,
            embeds_binding,
            context_tokens: prompt_tokens,
            prompt_len: 0,
            generated_count: 0,
        };
        backend.prompt_len = backend.context_tokens.len();
        let mut loop_state = DecodeLoopState::new(0, options.seed, options.top_logprobs);
        let result = run_decode_loop(
            &mut backend,
            &mut loop_state,
            &options,
            &chain,
            tokenizer,
            None,
            callback,
        )?;
        Ok((result, tensors))
    }

    /// Post-decode counterpart to [`synthesize`](Self::synthesize) for a
    /// nested-AR (multi-decoder TTS) pipeline: drive the outer/inner loops (which
    /// publish `{outer}.output_codes` into the pool), then run the `final_only`
    /// vocoder stage over the pool to produce the waveform.
    fn synthesize_nested(
        &mut self,
        pipeline_request: PipelineGenerateRequest,
    ) -> anyhow::Result<PipelineSynthesis> {
        let post_decode_components = match &self.plan {
            PipelinePlan::NestedAutoregressive(plan) => plan.post_decode_components.clone(),
            _ => anyhow::bail!("internal error: synthesize_nested on a non-nested plan"),
        };
        let (generation, mut tensors) = self.run_nested_autoregressive(pipeline_request)?;
        self.run_prompt_phase_components(&post_decode_components, &mut tensors, "postlogue", None)?;
        Ok(PipelineSynthesis {
            generation,
            tensors,
        })
    }

    /// Drive a **dual, hierarchically-nested autoregressive** pipeline — the
    /// multi-decoder TTS (Qwen3-TTS-style) shape (DESIGN.md §20.3).
    ///
    /// The **outer** decoder (talker) runs up to `max_frames` frames; each outer
    /// step (one audio frame) produces a `last_hidden_state` that seeds the
    /// **inner** decoder (code_predictor) AR loop of `num_code_groups` steps.
    /// The inner loop threads the outer hidden state at inner step 0 and the
    /// inner decoder's own per-code embedding output on later steps. Every code
    /// group is assembled into the synthetic pool tensor `{outer}.output_codes`
    /// of shape `[1, frames, num_code_groups]` (int64), and the flattened codes
    /// are returned as the [`GenerateResult`]'s token ids.
    fn run_nested_autoregressive(
        &mut self,
        pipeline_request: PipelineGenerateRequest,
    ) -> anyhow::Result<(GenerateResult, PipelineTensors)> {
        let plan = match &self.plan {
            PipelinePlan::NestedAutoregressive(plan) => plan.clone(),
            _ => anyhow::bail!(
                "synthesize()/generate() on a nested pipeline requires a nested_autoregressive plan"
            ),
        };

        let options = pipeline_request.request.options.clone();
        options.validate()?;
        let prompt_tokens = tokenize_with(self.tokenizer()?, &pipeline_request.request.prompt)?;
        if prompt_tokens.is_empty() {
            anyhow::bail!("prompt must contain at least one token");
        }

        let mut tensors = pipeline_request.inputs;
        self.seed_prompt_token_inputs(&plan.prompt_components, &prompt_tokens, &mut tensors)?;
        self.run_prompt_phase_components(&plan.prompt_components, &mut tensors, "prologue", None)?;

        // Fixed routed extras for each decoder (encoder conditioning etc.). The
        // inner decoder's seed input is threaded per inner step, so exclude it.
        // In pre-embedder mode the outer decoder's per-step `inputs_embeds` is
        // built each frame (not a fixed routed extra), so exclude it too.
        let outer_extra_exclude = plan.pre_embedder.as_ref().map(|p| p.outer_input.as_str());
        let outer_extras = self.decoder_extra_inputs(&plan.outer, &tensors, outer_extra_exclude)?;
        let inner_extras =
            self.decoder_extra_inputs(&plan.inner, &tensors, Some(&plan.inner_embeds_input))?;

        let outer_session = self
            .models
            .session(&plan.outer)
            .with_context(|| format!("nested outer decoder '{}' was not loaded", plan.outer))?;
        let inner_session = self
            .models
            .session(&plan.inner)
            .with_context(|| format!("nested inner decoder '{}' was not loaded", plan.inner))?;

        // Resolve the pre-embedder session and its input ports (sessions are not
        // available at plan-build time, so this is done here). `frame_codes` is
        // the sole int64 input (or one named `frame_codes`); the optional
        // `text_embed` trailing-text input is a second float input.
        let pre_embed = match plan.pre_embedder.as_ref() {
            Some(binding) => {
                let session = self.models.session(&binding.component).with_context(|| {
                    format!("nested pre_embedder '{}' was not loaded", binding.component)
                })?;
                let frame_codes_input = session
                    .inputs()
                    .iter()
                    .find(|info| info.name == "frame_codes")
                    .or_else(|| {
                        session
                            .inputs()
                            .iter()
                            .find(|info| info.dtype == DataType::Int64)
                    })
                    .map(|info| info.name.clone())
                    .with_context(|| {
                        format!(
                            "nested pre_embedder '{}' must expose an int64 'frame_codes' input",
                            binding.component
                        )
                    })?;
                let text_embed_input = session
                    .inputs()
                    .iter()
                    .find(|info| {
                        info.name != frame_codes_input
                            && (info.name == "text_embed"
                                || matches!(
                                    info.dtype,
                                    DataType::Float32 | DataType::Float16 | DataType::BFloat16
                                ))
                    })
                    .map(|info| info.name.clone());
                // Hidden size for the per-step embedding / zero `text_embed`:
                // prefer the outer decoder's `inputs_embeds` input, fall back to
                // the pre-embedder's `inputs_embeds` output.
                let hidden = outer_session
                    .inputs()
                    .iter()
                    .find(|info| info.name == binding.outer_input)
                    .and_then(|info| info.shape.last().copied())
                    .filter(|dim| *dim > 0)
                    .or_else(|| {
                        session
                            .outputs()
                            .iter()
                            .find(|info| info.name.to_ascii_lowercase().ends_with("inputs_embeds"))
                            .and_then(|info| info.shape.last().copied())
                            .filter(|dim| *dim > 0)
                    })
                    .map(|dim| dim as usize)
                    .with_context(|| {
                        format!(
                            "could not determine hidden size for nested pre_embedder '{}' \
                             (outer '{}' input '{}' has no static last dim)",
                            binding.component, plan.outer, binding.outer_input
                        )
                    })?;
                Some(ResolvedPreEmbedder {
                    session,
                    outer_input: binding.outer_input.clone(),
                    frame_codes_input,
                    text_embed_input,
                    hidden,
                })
            }
            None => None,
        };

        // The inner decoder's per-code embedding output: its sole output that is
        // neither logits nor a present-KV tensor. Threaded into the next inner
        // step's seed input.
        let inner_embed_output = inner_session
            .output_names()
            .iter()
            .find(|name| {
                let lower = name.to_ascii_lowercase();
                !lower.contains("logits") && !is_present_output(name)
            })
            .cloned()
            .with_context(|| {
                format!(
                    "nested inner decoder '{}' must expose a per-code embedding output (a \
                     non-logits, non-KV output) to thread across inner steps",
                    plan.inner
                )
            })?;

        let mut outer_state = DecodeState::new(outer_session)?;
        let mut codes: Vec<i64> = Vec::with_capacity(plan.max_frames * plan.num_code_groups);
        // The outer loop feeds the full prompt on frame 0 (prefill), then the
        // previous frame's outer argmax token on each subsequent frame.
        let mut outer_input_tokens = prompt_tokens.clone();
        let mut outer_past_len = 0usize;
        // Pre-embedder mode only: the previous frame's assembled code tuple
        // `[outer_code_0, inner_code_1, ..., inner_code_{G-1}]`, used to build the
        // next frame's `inputs_embeds`. `None` on frame 0 (prefill).
        let mut prev_frame_codes: Option<Vec<i64>> = None;

        for _frame in 0..plan.max_frames {
            // --- Outer talker step: one audio frame. ---
            let outer_outputs = if let Some(pre) = &pre_embed {
                // Build this frame's `frame_codes` from the previous frame's code
                // tuple (frame 0 uses a zero seed — real prompt-embeds prefill is
                // a follow-up), run the pre-embedder to materialize the talker's
                // `inputs_embeds`, and feed it as a single-position extra input.
                let frame_codes = prev_frame_codes
                    .clone()
                    .unwrap_or_else(|| vec![0i64; plan.num_code_groups]);
                let inputs_embeds = run_pre_embedder(pre, &frame_codes)?;
                let mut step_extras = Vec::with_capacity(outer_extras.len() + 1);
                for (name, value) in &outer_extras {
                    step_extras.push((name.clone(), clone_value(value)?));
                }
                step_extras.push((pre.outer_input.clone(), inputs_embeds));
                let outputs = run_decode_step_with_extra(
                    outer_session,
                    &mut outer_state,
                    &[0],
                    outer_past_len,
                    &step_extras,
                )?;
                outer_past_len += 1;
                outputs
            } else {
                let outputs = run_decode_step_with_extra(
                    outer_session,
                    &mut outer_state,
                    &outer_input_tokens,
                    outer_past_len,
                    &outer_extras,
                )?;
                outer_past_len += outer_input_tokens.len();
                outputs
            };

            let outer_logits = named_output(outer_session, &outer_outputs, "logits", true)?;
            let outer_token = argmax_last_row(outer_logits)?;
            let hidden = named_output(
                outer_session,
                &outer_outputs,
                &plan.outer_hidden_output,
                false,
            )?;
            let seed = last_position_hidden(hidden)?;
            // The talker autoregresses on its own per-frame prediction.
            outer_input_tokens = vec![u32::try_from(outer_token).unwrap_or(0)];

            // --- Inner code_predictor loop: num_code_groups residual codes. ---
            let mut inner_state = DecodeState::new(inner_session)?;
            let mut inner_embeds = seed;
            let mut frame_inner_codes: Vec<i64> = Vec::with_capacity(plan.num_code_groups);
            for step in 0..plan.num_code_groups {
                let mut step_extras = Vec::with_capacity(inner_extras.len() + 1);
                for (name, value) in &inner_extras {
                    step_extras.push((name.clone(), clone_value(value)?));
                }
                step_extras.push((plan.inner_embeds_input.clone(), inner_embeds));

                let inner_outputs = run_decode_step_with_extra(
                    inner_session,
                    &mut inner_state,
                    &[0],
                    step,
                    &step_extras,
                )?;
                let inner_logits = named_output(inner_session, &inner_outputs, "logits", true)?;
                let inner_code = argmax_last_row(inner_logits)?;
                codes.push(inner_code);
                frame_inner_codes.push(inner_code);
                // Thread the inner decoder's per-code embedding into the next step.
                inner_embeds = clone_value(named_output(
                    inner_session,
                    &inner_outputs,
                    &inner_embed_output,
                    false,
                )?)?;
            }

            // Pre-embedder mode: remember this frame's code tuple for the next
            // frame's `frame_codes`: the talker's own code as group 0 and the
            // inner residuals for groups 1..G-1 (matching the real Qwen3-TTS
            // layout where code_0 comes from the talker, not the code predictor).
            if pre_embed.is_some() {
                let mut tuple = Vec::with_capacity(plan.num_code_groups);
                tuple.push(outer_token);
                tuple.extend_from_slice(&frame_inner_codes[1..]);
                prev_frame_codes = Some(tuple);
            }
        }

        // Publish the assembled per-frame codes as `{outer}.output_codes`
        // [1, frames, num_code_groups] (int64) for the post-decode vocoder stage.
        let codes_endpoint = format!("{}.output_codes", plan.outer);
        let codes_value = Value::from_slice_i64(
            &codes,
            &[1, plan.max_frames as i64, plan.num_code_groups as i64],
        )
        .with_context(|| format!("failed to build generated-codes tensor '{codes_endpoint}'"))?;
        tensors.insert(codes_endpoint, codes_value);

        let token_ids: Vec<TokenId> = codes
            .iter()
            .map(|&c| u32::try_from(c).unwrap_or(0))
            .collect();
        let result = GenerateResult {
            text: String::new(),
            token_ids,
            finish_reason: crate::FinishReason::MaxTokens,
            prefix_cache_hit_len: 0,
            logprobs: None,
        };
        Ok((result, tensors))
    }

    pub fn spec(&self) -> &PipelineSpec {
        &self.models.directory.spec
    }

    /// The `init_noise_sigma` of the diffusion scheduler this pipeline drives.
    ///
    /// Returns `None` when the pipeline is not iterative (diffusion) or carries
    /// no scheduler. The caller pre-scales the seed latent by this factor so it
    /// lives in the scheduler's sigma space (`1.0` for DDIM / DPM-Solver++;
    /// `sigmas[0]` for Euler / Euler-Ancestral). This lets a runner reuse the
    /// exact scheduler the pipeline builds instead of duplicating the sigma math.
    pub fn diffusion_init_noise_sigma(&self) -> Option<f32> {
        match &self.plan {
            PipelinePlan::Iterative(iterative) => iterative
                .scheduler
                .as_ref()
                .map(|scheduler| scheduler.init_noise_sigma()),
            _ => None,
        }
    }

    /// Execute a **non-autoregressive** pipeline (single-pass or iterative /
    /// diffusion) and return the final named output tensors, keyed by
    /// `component.output_name`.
    ///
    /// This is the tensor-producing counterpart to [`generate`](Self::generate)
    /// (which drives an autoregressive token loop). Use it for diffusion
    /// denoisers, VAE decoders, audio vocoders, and other tensor-out models.
    pub fn run_pipeline(
        &mut self,
        request: PipelineGenerateRequest,
    ) -> anyhow::Result<PipelineTensors> {
        match &self.plan {
            PipelinePlan::Iterative(_) => self.run_iterative(request),
            PipelinePlan::SinglePass(_) => self.run_single_pass(request),
            PipelinePlan::Composite(_) => self.run_composite(request),
            PipelinePlan::Autoregressive(_) => anyhow::bail!(
                "run_pipeline() runs single-pass or iterative pipelines; use generate() for \
                 autoregressive text pipelines"
            ),
            PipelinePlan::NestedAutoregressive(_) => anyhow::bail!(
                "run_pipeline() runs single-pass or iterative pipelines; use synthesize() for \
                 a nested-autoregressive (multi-decoder TTS) pipeline"
            ),
        }
    }

    /// Run a bounded iterative (diffusion) denoise loop.
    ///
    /// Semantics: prompt-phase components run once; then the denoiser runs
    /// `num_steps` times, threading loop-carried state (its self-edges) from one
    /// step's output into the next step's input while constant conditioning
    /// (e.g. encoder hidden states) is re-supplied each step; then final-phase
    /// components run once. `guidance_scale` is carried but not yet applied —
    /// classifier-free guidance and timestep/sigma schedules are supplied by the
    /// scheduler-registry follow-up.
    fn run_iterative(
        &self,
        request: PipelineGenerateRequest,
    ) -> anyhow::Result<PipelineTensors> {
        let PipelinePlan::Iterative(plan) = &self.plan else {
            anyhow::bail!("internal error: run_iterative on a non-iterative plan");
        };

        // Live overrides (ComfyUI-style): re-drive the already-loaded models with
        // different loop parameters, no reload. Seed / prompt / negative are
        // already live via per-request inputs, so only loop params are overridden.
        let overrides = &request.iterative_overrides;
        let num_steps = overrides.num_steps.unwrap_or(plan.num_steps);
        let start_step = overrides.start_step.unwrap_or(plan.start_step);
        if num_steps == 0 {
            anyhow::bail!("iterative override num_steps must be >= 1");
        }
        if start_step >= num_steps {
            anyhow::bail!(
                "iterative override start_step ({start_step}) must be < num_steps ({num_steps})"
            );
        }
        // Rebuild the scheduler when the step count changes (its schedule may be
        // baked at build time). An explicit per-step timestep schedule is tied to
        // the original step count, so reject a step-count override in that case.
        let rebuilt_scheduler = if num_steps != plan.num_steps {
            if plan.timesteps.is_some() {
                anyhow::bail!(
                    "cannot override num_steps for a pipeline with an explicit timestep schedule"
                );
            }
            match &plan.scheduler_spec {
                Some(spec) => Some(plan.scheduler_registry.build(spec, num_steps)?),
                None => None,
            }
        } else {
            None
        };
        let scheduler = rebuilt_scheduler.as_ref().or(plan.scheduler.as_ref());

        // Classifier-free guidance scale (active only when set and != 1.0).
        let guidance = overrides
            .guidance_scale
            .or(plan.guidance_scale)
            .filter(|s| *s != 1.0);
        // `constants` holds external inputs + prompt-phase outputs and is NOT
        // mutated by the loop, so a denoiser whose output port shares a name
        // with a conditioning input cannot clobber that conditioning. Denoiser
        // outputs live in a separate `loop_state`, keyed by output port.
        let mut constants = request.inputs;
        let mut stage_timings: Vec<serde_json::Value> = Vec::new();
        self.run_prompt_phase_components(
            &plan.prompt_components,
            &mut constants,
            "encode",
            Some(&mut stage_timings),
        )?;

        let denoiser = self
            .models
            .session(&plan.denoiser)
            .with_context(|| format!("pipeline denoiser '{}' was not loaded", plan.denoiser))?;

        // Precompute the CFG unconditional conditioning once. Any denoiser input
        // port with a supplied `{denoiser}.{port}.uncond` embedding is overridden
        // on the unconditional pass — this supports multi-conditioning models
        // (e.g. SDXL overrides both `encoder_hidden_states` and pooled
        // `text_embeds`, while sharing `time_ids`). The primary
        // `cfg_conditioning_input` is additionally zeroed when no `.uncond` is
        // supplied (the zeros fallback for a single-conditioning SD model).
        let cfg_uncond: Vec<(String, Value)> = if guidance.is_some() {
            if let Some(primary) = plan.cfg_conditioning_input.clone() {
                let mut overrides: Vec<(String, Value)> = Vec::new();
                let mut seen: BTreeSet<String> = BTreeSet::new();
                for info in denoiser.inputs() {
                    let port = info.name.as_str();
                    let uncond_endpoint = format!("{}.{}.uncond", plan.denoiser, port);
                    if let Some(u) = constants.get(&uncond_endpoint) {
                        overrides.push((port.to_string(), clone_value(u)?));
                        seen.insert(port.to_string());
                    }
                }
                if !seen.contains(&primary) {
                    let cond_endpoint = format!("{}.{}", plan.denoiser, primary);
                    let cond = constants
                        .get(&cond_endpoint)
                        .or_else(|| {
                            plan.dataflow
                                .iter()
                                .find(|e| e.to == cond_endpoint)
                                .and_then(|e| constants.get(&e.from))
                        })
                        .with_context(|| format!("cfg conditioning '{cond_endpoint}' not found"))?;
                    overrides.push((
                        primary.clone(),
                        Value::from_slice_f32(&vec![0.0f32; cond.numel()], cond.shape())?,
                    ));
                }
                overrides
            } else {
                // No static conditioning input: the unconditional pass is a
                // transform of the loop-carried sample (discrete language
                // diffusion re-masks the prompt via `cfg_uncond_sample`).
                Vec::new()
            }
        } else {
            Vec::new()
        };

        // `carried` holds the value to feed each loop-carried INPUT port next
        // step (keyed by input port); `last_outputs` holds the denoiser's raw
        // outputs from the final step (keyed by output port). Keeping them
        // separate from the immutable `constants` pool prevents an output whose
        // name collides with a conditioning input from clobbering it.
        let mut carried: HashMap<String, Value> = HashMap::new();
        let mut last_outputs: HashMap<String, Value> = HashMap::new();
        // Reset any multistep scheduler state before the loop (img2img reuses a
        // plan whose scheduler may hold state from a previous run).
        if let Some(scheduler) = scheduler {
            scheduler.reset();
        }
        // Denoiser timestep schedule: prefer the plan's explicit `strategy.timesteps`,
        // otherwise fall back to the scheduler's own timesteps (so from-scratch
        // packages that omit the table still drive the denoiser with the correct
        // diffusion timesteps rather than the raw step index).
        let scheduler_timesteps: Option<Vec<f32>> = if plan.timesteps.is_some() {
            None
        } else {
            scheduler.and_then(|scheduler| scheduler.timesteps())
        };
        // Partial (img2img) loops start at `start_step`; the seed is then the
        // encoded image already noised to `timesteps[start_step]`.
        let denoise_start = std::time::Instant::now();
        for step in start_step..num_steps {
            let step_start = std::time::Instant::now();
            let is_first = step == start_step;
            // Timestep/sigma for this step: explicit plan schedule when provided,
            // else the scheduler's timesteps, else the 0-based step index.
            let timestep = plan
                .timesteps
                .as_ref()
                .or(scheduler_timesteps.as_ref())
                .and_then(|ts| ts.get(step).copied())
                .unwrap_or(step as f32);

            // Raw (unscaled) loop-carried sample feeding each loop input this
            // step: the seed on the first step, otherwise the value carried from
            // the previous step. The scheduler's `step` consumes these raw samples.
            let mut raw_samples: HashMap<String, Value> = HashMap::new();
            for (_, in_port) in &plan.loop_edges {
                let raw = if is_first {
                    let endpoint = format!("{}.{}", plan.denoiser, in_port);
                    constants.get(&endpoint).with_context(|| {
                        format!("missing iterative pipeline seed '{endpoint}' at start step")
                    })?
                } else {
                    carried.get(in_port).with_context(|| {
                        format!("loop-carried input '{}.{in_port}' was not produced", plan.denoiser)
                    })?
                };
                raw_samples.insert(in_port.clone(), clone_value(raw)?);
            }

            // Some schedulers (e.g. Euler) scale the loop-carried sample before
            // it reaches the denoiser. Compute those scaled values once and feed
            // them as per-port overrides; schedulers that don't scale (DDIM,
            // masked diffusion) leave the raw sample untouched.
            let mut scaled_inputs: HashMap<String, Value> = HashMap::new();
            if let Some(scheduler) = scheduler {
                for (_, in_port) in &plan.loop_edges {
                    let raw = &raw_samples[in_port];
                    if let Some(scaled) = scheduler.scale_input(step, num_steps, raw)? {
                        scaled_inputs.insert(in_port.clone(), scaled);
                    }
                }
            }
            let scale_overrides: Vec<(&str, &Value)> = scaled_inputs
                .iter()
                .map(|(port, value)| (port.as_str(), value))
                .collect();

            // Conditional pass (all inputs as declared, plus any input scaling).
            let cond_out = self.run_denoiser_pass(
                denoiser,
                plan,
                start_step,
                &constants,
                &carried,
                step,
                timestep,
                &scale_overrides,
            )?;

            // Classifier-free guidance: run an unconditional pass with the
            // conditioning replaced by the unconditional embedding, then combine
            // per output port:  pred = uncond + scale * (cond - uncond).
            let out_map = if let Some(scale) = guidance {
                let mut cfg_overrides = scale_overrides.clone();
                for (port, value) in &cfg_uncond {
                    cfg_overrides.retain(|(p, _)| *p != port.as_str());
                    cfg_overrides.push((port.as_str(), value));
                }
                // Language-diffusion CFG: the unconditional pass feeds the
                // loop-carried input with its prompt tokens re-masked. Computed
                // per step from the current sample (owned here so its references
                // live through the unconditional denoiser pass).
                let mut prompt_masked_inputs: Vec<(String, Value)> = Vec::new();
                if let Some(scheduler) = scheduler {
                    for (_, in_port) in &plan.loop_edges {
                        let raw = &raw_samples[in_port];
                        if let Some(uncond_sample) = scheduler.cfg_uncond_sample(raw)? {
                            prompt_masked_inputs.push((in_port.clone(), uncond_sample));
                        }
                    }
                }
                for (port, value) in &prompt_masked_inputs {
                    cfg_overrides.retain(|(p, _)| *p != port.as_str());
                    cfg_overrides.push((port.as_str(), value));
                }
                let uncond_out = self.run_denoiser_pass(
                    denoiser,
                    plan,
                    start_step,
                    &constants,
                    &carried,
                    step,
                    timestep,
                    &cfg_overrides,
                )?;
                let mut combined: HashMap<String, Value> = HashMap::new();
                for (port, cond_value) in &cond_out {
                    let uncond_value = uncond_out.get(port).with_context(|| {
                        format!("unconditional pass did not produce '{}.{port}'", plan.denoiser)
                    })?;
                    let cond_v = cond_value.to_vec_f32_lossy()?;
                    let uncond_v = uncond_value.to_vec_f32_lossy()?;
                    let guided: Vec<f32> = uncond_v
                        .iter()
                        .zip(&cond_v)
                        .map(|(u, c)| u + scale * (c - u))
                        .collect();
                    combined.insert(port.clone(), Value::from_slice_f32(&guided, cond_value.shape())?);
                }
                combined
            } else {
                cond_out
            };

            // Compute the next value for each loop-carried input. Without a
            // scheduler this is identity feedback (output -> input). With a
            // scheduler the output is a noise prediction and the next sample is
            // `scheduler.step(raw_sample, prediction)` (raw = unscaled).
            for (out_port, in_port) in &plan.loop_edges {
                let model_output = out_map.get(out_port).with_context(|| {
                    format!("denoiser did not produce loop output '{}.{out_port}'", plan.denoiser)
                })?;
                let next = if let Some(scheduler) = scheduler {
                    let sample = raw_samples.get(in_port).with_context(|| {
                        format!("missing loop-carried sample for '{}.{in_port}'", plan.denoiser)
                    })?;
                    if scheduler.needs_noise() {
                        let noise = self.step_noise(plan, num_steps, &constants, in_port, step, sample)?;
                        scheduler.step_with_noise(
                            step,
                            num_steps,
                            sample,
                            model_output,
                            Some(&noise),
                        )?
                    } else {
                        scheduler.step(step, num_steps, sample, model_output)?
                    }
                } else {
                    clone_value(model_output)?
                };
                dump_iterative_step(
                    &plan.denoiser,
                    in_port,
                    step,
                    &next,
                    step_start.elapsed().as_secs_f64() * 1e3,
                );
                carried.insert(in_port.clone(), next);
            }
            last_outputs = out_map;
        }
        let denoise_ms = denoise_start.elapsed().as_secs_f64() * 1e3;
        stage_timings.push(serde_json::json!({
            "component": plan.denoiser,
            "phase": "denoise",
            "ms": denoise_ms,
            "steps": num_steps - start_step,
        }));

        // Publish the final denoiser outputs (raw predictions) and the final
        // loop-carried samples, then run final-phase components once. A VAE can
        // route from either the output port or the (post-scheduler) sample port.
        let mut tensors = constants;
        for (out_port, value) in last_outputs {
            tensors.insert(format!("{}.{}", plan.denoiser, out_port), value);
        }
        for (in_port, value) in carried {
            tensors.insert(format!("{}.{}", plan.denoiser, in_port), value);
        }
        self.run_prompt_phase_components(
            &plan.final_components,
            &mut tensors,
            "decode",
            Some(&mut stage_timings),
        )?;
        dump_stage_timings(&stage_timings);
        Ok(tensors)
    }

    /// Run one denoiser invocation for `step`. Returns `(outputs, sample_in)`
    /// keyed by port. `override_input`, when set as `(port, value)`, substitutes
    /// that input's value — used to supply the unconditional conditioning on the
    /// CFG unconditional pass.
    #[allow(clippy::too_many_arguments)]
    fn run_denoiser_pass(
        &self,
        denoiser: &Session,
        plan: &IterativePlan,
        start_step: usize,
        constants: &PipelineTensors,
        carried: &HashMap<String, Value>,
        step: usize,
        timestep: f32,
        overrides: &[(&str, &Value)],
    ) -> anyhow::Result<HashMap<String, Value>> {
        let mut inputs: Vec<(String, Value)> = Vec::new();
        for info in denoiser.inputs() {
            let port = info.name.as_str();
            let endpoint = format!("{}.{}", plan.denoiser, port);
            // An override wins for its port. Two producers use overrides: the
            // scheduler's per-step input scaling (Euler) and CFG's unconditional
            // conditioning embedding.
            if let Some((_, over_value)) = overrides.iter().find(|(p, _)| *p == port) {
                inputs.push((port.to_string(), coerce_value_to_dtype(over_value, info.dtype)?));
                continue;
            }
            // Per-step timestep injection takes precedence for its port. Honor
            // the port dtype: real diffusion denoisers (DiT/UNet) declare an
            // INT64 timestep, while others take a float sigma.
            if plan.timestep_input.as_deref() == Some(port) {
                let ts = match info.dtype {
                    DataType::Int64 => Value::from_vec_i64(vec![timestep as i64], &[1])?,
                    _ => Value::from_slice_f32(&[timestep], &[1])?,
                };
                inputs.push((port.to_string(), ts));
                continue;
            }
            let is_loop = plan.loop_edges.iter().any(|(_, in_port)| in_port == port);
            let value = if is_loop {
                if step == start_step {
                    constants.get(&endpoint).with_context(|| {
                        format!("missing iterative pipeline seed '{endpoint}' at start step")
                    })?
                } else {
                    carried
                        .get(port)
                        .with_context(|| format!("loop-carried input '{endpoint}' was not produced"))?
                }
            } else {
                let routed = plan
                    .dataflow
                    .iter()
                    .find(|edge| edge.to == endpoint)
                    .and_then(|edge| constants.get(&edge.from));
                constants
                    .get(&endpoint)
                    .or(routed)
                    .with_context(|| format!("missing pipeline input '{endpoint}'"))?
            };
            inputs.push((port.to_string(), coerce_value_to_dtype(value, info.dtype)?));
        }
        let refs = inputs
            .iter()
            .map(|(name, value)| (name.as_str(), value))
            .collect::<Vec<_>>();
        let outputs = denoiser.run(&refs).map_err(|e| {
            anyhow::anyhow!("ORT denoiser '{}' failed at step {step}: {e}", plan.denoiser)
        })?;
        let mut out_map: HashMap<String, Value> = HashMap::new();
        for (name, value) in denoiser.output_names().iter().zip(outputs) {
            out_map.insert(name.clone(), value);
        }
        Ok(out_map)
    }

    /// Fetch the per-step Gaussian noise an ancestral scheduler needs at `step`.
    ///
    /// The caller supplies an external tensor `{denoiser}.{in_port}.noise` shaped
    /// `[num_steps, *sample_shape]` (so the noise sequence is reproducible and can
    /// match a reference generator); this slices out the `step`-th sample.
    fn step_noise(
        &self,
        plan: &IterativePlan,
        num_steps: usize,
        constants: &PipelineTensors,
        in_port: &str,
        step: usize,
        sample: &Value,
    ) -> anyhow::Result<Value> {
        let endpoint = format!("{}.{}.noise", plan.denoiser, in_port);
        let all = constants.get(&endpoint).with_context(|| {
            format!(
                "ancestral scheduler requires per-step noise tensor '{endpoint}' \
                 shaped [num_steps, ...]"
            )
        })?;
        let elem: usize = sample.shape().iter().map(|&d| d as usize).product();
        let data = all.to_vec_f32_lossy()?;
        let want = num_steps * elem;
        if data.len() != want {
            anyhow::bail!(
                "noise tensor '{endpoint}' has {} elements but expected {want} \
                 ({num_steps} steps x {elem})",
                data.len(),
            );
        }
        let slice = &data[step * elem..(step + 1) * elem];
        Value::from_slice_f32(slice, sample.shape()).map_err(Into::into)
    }

    /// Run a single-pass pipeline: prompt-phase components once, then one
    /// forward invocation of the strategy `model`.
    /// Execute a multi-stage composite pipeline (DESIGN.md §20): run each stage
    /// once, in declared order, over a shared tensor pool. A stage's model reads
    /// its inputs from the pool (routed by the pipeline dataflow) and writes its
    /// outputs back, so an earlier stage's outputs feed later stages.
    fn run_composite(
        &self,
        request: PipelineGenerateRequest,
    ) -> anyhow::Result<PipelineTensors> {
        let PipelinePlan::Composite(plan) = &self.plan else {
            anyhow::bail!("internal error: run_composite on a non-composite plan");
        };
        let mut tensors = request.inputs;
        for stage in &plan.stages {
            match &stage.kind {
                CompositeStageKind::SinglePass { model } => {
                    self.run_prompt_phase_components(
                        std::slice::from_ref(model),
                        &mut tensors,
                        &stage.name,
                        None,
                    )?;
                }
            }
        }
        Ok(tensors)
    }

    fn run_single_pass(
        &self,
        request: PipelineGenerateRequest,
    ) -> anyhow::Result<PipelineTensors> {
        let PipelinePlan::SinglePass(plan) = &self.plan else {
            anyhow::bail!("internal error: run_single_pass on a non-single-pass plan");
        };
        let mut tensors = request.inputs;
        self.run_prompt_phase_components(&plan.prompt_components, &mut tensors, "prologue", None)?;

        let model = self
            .models
            .session(&plan.model)
            .with_context(|| format!("pipeline model '{}' was not loaded", plan.model))?;
        let inputs = self.component_inputs(&plan.model, model, &tensors)?;
        let refs = inputs
            .iter()
            .map(|(name, value)| (name.as_str(), value))
            .collect::<Vec<_>>();
        let outputs = model
            .run(&refs)
            .map_err(|e| anyhow::anyhow!("ORT pipeline model '{}' failed: {e}", plan.model))?;
        for (name, value) in model.output_names().iter().zip(outputs) {
            tensors.insert(format!("{}.{}", plan.model, name), value);
        }
        Ok(tensors)
    }

    fn tokenizer(&self) -> anyhow::Result<&Tokenizer> {
        self.models
            .tokenizer_for(&self.tokenizer_component)
            .with_context(|| format!("no tokenizer available for '{}'", self.tokenizer_component))
    }

    fn run_prompt_phase_components(
        &self,
        components: &[String],
        tensors: &mut PipelineTensors,
        phase: &str,
        mut timings: Option<&mut Vec<serde_json::Value>>,
    ) -> anyhow::Result<()> {
        for component in components {
            let session = self
                .models
                .session(component)
                .with_context(|| format!("pipeline component '{component}' was not loaded"))?;
            let inputs = self.component_inputs(component, session, tensors)?;
            let refs = inputs
                .iter()
                .map(|(name, value)| (name.as_str(), value))
                .collect::<Vec<_>>();
            let started = std::time::Instant::now();
            let outputs = session
                .run(&refs)
                .map_err(|e| anyhow::anyhow!("ORT pipeline component '{component}' failed: {e}"))?;
            if let Some(sink) = timings.as_deref_mut() {
                sink.push(serde_json::json!({
                    "component": component,
                    "phase": phase,
                    "ms": started.elapsed().as_secs_f64() * 1e3,
                }));
            }
            for (name, value) in session.output_names().iter().zip(outputs) {
                tensors.insert(format!("{component}.{name}"), value);
            }
        }
        Ok(())
    }

    fn component_inputs(
        &self,
        component: &str,
        session: &Session,
        tensors: &PipelineTensors,
    ) -> anyhow::Result<Vec<(String, Value)>> {
        let mut inputs = Vec::new();
        for info in session.inputs() {
            let endpoint = format!("{component}.{}", info.name);
            let routed = self
                .plan
                .dataflow()
                .iter()
                .find(|edge| edge.to == endpoint)
                .and_then(|edge| tensors.get(&edge.from));
            let value = tensors
                .get(&endpoint)
                .or(routed)
                .with_context(|| format!("missing pipeline input '{endpoint}'"))?;
            inputs.push((info.name.clone(), coerce_value_to_dtype(value, info.dtype)?));
        }
        Ok(inputs)
    }

    fn decoder_extra_inputs(
        &self,
        decoder: &str,
        tensors: &PipelineTensors,
        exclude_input: Option<&str>,
    ) -> anyhow::Result<Vec<(String, Value)>> {
        let mut extras = Vec::new();
        for edge in self
            .plan
            .edges_to_component(decoder)
            .filter(|edge| endpoint_component(&edge.from).is_some_and(|from| from != decoder))
        {
            let (_, input) = parse_endpoint(&edge.to)?;
            // The per-step `inputs_embeds` edge is threaded dynamically by the
            // decode loop (re-embedding each step), not carried as a fixed extra.
            if exclude_input == Some(input) {
                continue;
            }
            let value = tensors
                .get(&edge.from)
                .with_context(|| format!("missing routed pipeline tensor '{}'", edge.from))?;
            extras.push((input.to_string(), clone_value(value)?));
        }
        Ok(extras)
    }

    /// Seed the prompt token ids into the shared pool for any prompt-phase
    /// component that consumes a token input (`input_ids`) which is neither
    /// supplied by the caller nor routed by a dataflow edge.
    ///
    /// A Gemma4-style VLM fuses image features into text-token embeddings via a
    /// prompt-phase `embedding` component whose `input_ids` come from the prompt
    /// itself (not from another model). Without this seam that component would
    /// fail with a `missing pipeline input 'embedding.input_ids'` error.
    fn seed_prompt_token_inputs(
        &self,
        components: &[String],
        prompt_tokens: &[TokenId],
        tensors: &mut PipelineTensors,
    ) -> anyhow::Result<()> {
        for component in components {
            let session = self
                .models
                .session(component)
                .with_context(|| format!("pipeline component '{component}' was not loaded"))?;
            for info in session.inputs() {
                if !is_token_input_name(&info.name.to_ascii_lowercase()) {
                    continue;
                }
                let endpoint = format!("{component}.{}", info.name);
                let routed = self.plan.dataflow().iter().any(|edge| edge.to == endpoint);
                if routed || tensors.contains_key(&endpoint) {
                    continue;
                }
                let ids: Vec<i64> = prompt_tokens.iter().map(|&t| i64::from(t)).collect();
                let value = Value::from_slice_i64(&ids, &[1, ids.len() as i64])?;
                tensors.insert(endpoint, value);
            }
        }
        Ok(())
    }

    /// Detect a decoder whose per-step sequence input is `inputs_embeds` (fused
    /// image + text embeddings) rather than `input_ids`, and bind the fusion
    /// component that must re-embed the running token on every decode step.
    ///
    /// Returns `None` for a conventional decoder that carries its own token
    /// input (and embeds internally). When `Some`, the decode loop re-runs the
    /// fusion component each step with the running token so the decoder receives
    /// a single-token `inputs_embeds`; cross-conditioning inputs (image features)
    /// are resolved once and re-supplied unchanged.
    fn embeds_step_binding(
        &self,
        decoder: &str,
        tensors: &PipelineTensors,
    ) -> anyhow::Result<Option<EmbedsStepBinding>> {
        let session = self
            .models
            .session(decoder)
            .with_context(|| format!("pipeline decoder '{decoder}' was not loaded"))?;
        // A decoder with its own token input embeds internally: nothing to bind.
        if session
            .inputs()
            .iter()
            .any(|info| is_token_input_name(&info.name.to_ascii_lowercase()))
        {
            return Ok(None);
        }
        let Some(embeds_input) = session.inputs().iter().find(|info| {
            let lower = info.name.to_ascii_lowercase();
            lower == "inputs_embeds" || lower.ends_with(".inputs_embeds")
        }) else {
            return Ok(None);
        };
        let decoder_input = embeds_input.name.clone();
        let endpoint = format!("{decoder}.{decoder_input}");
        let edge = self
            .plan
            .dataflow()
            .iter()
            .find(|edge| edge.to == endpoint)
            .with_context(|| {
                format!(
                    "decoder '{decoder}' consumes '{decoder_input}' but no dataflow edge feeds \
                     it; a Gemma4-style VLM needs an embedding fusion component routed to \
                     '{endpoint}'"
                )
            })?;
        let (component, output) = parse_endpoint(&edge.from)?;
        let component = component.to_string();
        let output = output.to_string();
        let embed_session = self.models.session(&component).with_context(|| {
            format!("pipeline embedding component '{component}' was not loaded")
        })?;

        // The fusion component's token input carries the running token each step;
        // every other input (e.g. image_features) is fixed conditioning resolved
        // once here from the shared pool (directly or via a dataflow edge).
        let mut token_input = None;
        let mut conditioning = Vec::new();
        for info in embed_session.inputs() {
            if is_token_input_name(&info.name.to_ascii_lowercase()) {
                token_input = Some(info.name.clone());
                continue;
            }
            let cond_endpoint = format!("{component}.{}", info.name);
            let routed = self
                .plan
                .dataflow()
                .iter()
                .find(|edge| edge.to == cond_endpoint)
                .and_then(|edge| tensors.get(&edge.from));
            let value = tensors
                .get(&cond_endpoint)
                .or(routed)
                .with_context(|| format!("missing embedding fusion input '{cond_endpoint}'"))?;
            conditioning.push((info.name.clone(), coerce_value_to_dtype(value, info.dtype)?));
        }
        let token_input = token_input.with_context(|| {
            format!(
                "embedding fusion component '{component}' must consume a token input (input_ids) \
                 so the decode loop can embed the running token each step"
            )
        })?;
        let prefill = clone_value(tensors.get(&edge.from).with_context(|| {
            format!(
                "prompt-phase embeddings '{}' were not produced; the fusion component must run \
                 in the prompt phase to seed prefill",
                edge.from
            )
        })?)?;

        Ok(Some(EmbedsStepBinding {
            decoder_input,
            component,
            output,
            token_input,
            conditioning,
            prefill,
        }))
    }
}

fn tokenize_with(tokenizer: &Tokenizer, prompt: &GeneratePrompt) -> anyhow::Result<Vec<TokenId>> {
    match prompt {
        GeneratePrompt::TokenIds(tokens) => Ok(tokens.clone()),
        GeneratePrompt::Text(text) => tokenizer
            .encode(text)
            .map_err(|e| anyhow::anyhow!("Failed to tokenize prompt: {}", e)),
    }
}

/// Expand the single image placeholder token in `prompt_tokens` into
/// `tokens_per_tile * num_tiles` copies of that same token.
///
/// Returns the input unchanged when `num_image_tiles` is `None`.
///
/// Only a **single** placeholder occurrence is supported. `num_image_tiles` is
/// an aggregate tile count across all images in the request, so expanding
/// multiple placeholders by that aggregate would produce the wrong number of
/// image-token slots. Richer per-image metadata (and row/column separator
/// tokens) requires the full preprocessing path; this count-based path targets
/// separator-free single-image models only.
///
/// Errors when:
/// - `num_image_tiles` is `Some` but the pipeline metadata declares no vision
///   contract (`image_placeholder_token_id` or `tokens_per_tile` missing).
/// - The placeholder token ID does not fit in `TokenId` (u32).
/// - `tokens_per_tile` is zero.
/// - The prompt contains no placeholder token, or more than one.
/// - Arithmetic would overflow, or the expanded sequence is empty.
fn expand_image_placeholders_count_based(
    prompt_tokens: Vec<TokenId>,
    num_image_tiles: Option<usize>,
    vision: Option<&PipelineVisionConfig>,
) -> anyhow::Result<Vec<TokenId>> {
    let num_tiles = match num_image_tiles {
        None => return Ok(prompt_tokens),
        Some(n) => n,
    };

    let (placeholder_i64, tokens_per_tile) = match vision {
        Some(v) => match (v.image_placeholder_token_id, v.tokens_per_tile) {
            (Some(id), Some(tpt)) => (id, tpt),
            _ => anyhow::bail!(
                "image tile count supplied but pipeline metadata vision contract is incomplete: \
                 both image_placeholder_token_id and tokens_per_tile must be set"
            ),
        },
        None => anyhow::bail!(
            "image tile count supplied but pipeline metadata declares no vision section; \
             add pipeline.vision with image_placeholder_token_id and tokens_per_tile"
        ),
    };

    if tokens_per_tile == 0 {
        anyhow::bail!(
            "pipeline metadata tokens_per_tile is 0; must be at least 1"
        );
    }

    let placeholder_id: TokenId = u32::try_from(placeholder_i64)
        .with_context(|| {
            format!(
                "image_placeholder_token_id {placeholder_i64} is out of range for token ID (u32)"
            )
        })?;

    let placeholder_count = prompt_tokens
        .iter()
        .filter(|&&t| t == placeholder_id)
        .count();
    if placeholder_count == 0 {
        anyhow::bail!(
            "num_image_tiles supplied but prompt contains no image placeholder token \
             (id={placeholder_id}); the prompt must contain exactly one placeholder"
        );
    }
    if placeholder_count > 1 {
        anyhow::bail!(
            "multi-image count-based expansion is not supported: found {placeholder_count} image \
             placeholders (id={placeholder_id}) but only an aggregate tile count is available; \
             supply a single image or thread per-image tile counts"
        );
    }

    let expansion: usize = tokens_per_tile
        .checked_mul(num_tiles)
        .context("image token expansion overflow: tokens_per_tile * num_image_tiles is too large")?;

    // The single placeholder expands to `expansion` copies; all other tokens are kept.
    let new_len = prompt_tokens
        .len()
        .checked_sub(1)
        .and_then(|base| base.checked_add(expansion))
        .context("expanded prompt token sequence length overflows")?;

    let mut expanded = Vec::new();
    expanded
        .try_reserve_exact(new_len)
        .context("failed to allocate expanded prompt token sequence")?;

    for token in prompt_tokens {
        if token == placeholder_id {
            for _ in 0..expansion {
                expanded.push(placeholder_id);
            }
        } else {
            expanded.push(token);
        }
    }

    if expanded.is_empty() {
        anyhow::bail!(
            "image placeholder expansion produced an empty token sequence; \
             check that num_image_tiles > 0 and the prompt contains non-placeholder tokens"
        );
    }

    Ok(expanded)
}

/// Per-step embedding fusion for a decoder whose prompt input is `inputs_embeds`
/// (Gemma4-style VLM) rather than `input_ids`. Built by
/// [`PipelineEngine::embeds_step_binding`].
///
/// The decoder has no token input of its own, so each decode step re-runs the
/// fusion component to embed the running token into a single-token
/// `inputs_embeds`. The prompt (prefill) step reuses the full-prompt embeddings
/// already produced in the prompt phase.
struct EmbedsStepBinding {
    /// Decoder input port that receives the fused embeddings (`inputs_embeds`).
    decoder_input: String,
    /// Prompt-phase fusion component that produces the embeddings.
    component: String,
    /// The fusion component's output port routed to `decoder_input`.
    output: String,
    /// The fusion component's token input port (`input_ids`).
    token_input: String,
    /// Fusion conditioning (e.g. `image_features`) resolved once from the pool
    /// and re-supplied unchanged on every decode step.
    conditioning: Vec<(String, Value)>,
    /// Full-prompt embeddings from the prompt phase, reused for the prefill step.
    prefill: Value,
}

struct PipelineDecodeLoopBackend<'a> {
    decoder: &'a Session,
    decoder_state: &'a mut DecodeState,
    decoder_extras: &'a [(String, Value)],
    /// Fusion component session, present iff `embeds_binding` is `Some`.
    embed_session: Option<&'a Session>,
    /// Per-step `inputs_embeds` fusion binding for a Gemma4-style VLM decoder.
    embeds_binding: Option<EmbedsStepBinding>,
    context_tokens: Vec<TokenId>,
    prompt_len: usize,
    generated_count: usize,
}

impl PipelineDecodeLoopBackend<'_> {
    /// Build this step's decoder extra inputs. When an [`EmbedsStepBinding`] is
    /// active, append a freshly-computed `inputs_embeds`: the full-prompt
    /// embeddings on prefill, or the re-embedded running token on later steps.
    fn step_extras(&self) -> anyhow::Result<Vec<(String, Value)>> {
        let Some(binding) = &self.embeds_binding else {
            return self
                .decoder_extras
                .iter()
                .map(|(name, value)| Ok((name.clone(), clone_value(value)?)))
                .collect();
        };
        let embeds = if self.generated_count == 0 {
            clone_value(&binding.prefill)?
        } else {
            // Re-embed only the running (last generated) token. It is text-only
            // with no image placeholder, so the fusion yields the pure token
            // embedding; image conditioning is re-supplied but contributes zero.
            let embed_session = self
                .embed_session
                .expect("embeds binding implies a loaded fusion session");
            let last = *self
                .context_tokens
                .last()
                .expect("a decode step always has a prior token");
            let mut inputs: Vec<(String, Value)> =
                Vec::with_capacity(binding.conditioning.len() + 1);
            inputs.push((
                binding.token_input.clone(),
                Value::from_slice_i64(&[i64::from(last)], &[1, 1])?,
            ));
            for (name, value) in &binding.conditioning {
                inputs.push((name.clone(), clone_value(value)?));
            }
            let refs = inputs
                .iter()
                .map(|(name, value)| (name.as_str(), value))
                .collect::<Vec<_>>();
            let outputs = embed_session.run(&refs).map_err(|e| {
                anyhow::anyhow!(
                    "ORT embedding fusion '{}' failed during decode: {e}",
                    binding.component
                )
            })?;
            let index = embed_session
                .output_names()
                .iter()
                .position(|name| name == &binding.output)
                .with_context(|| {
                    format!(
                        "embedding fusion component '{}' has no output '{}'",
                        binding.component, binding.output
                    )
                })?;
            clone_value(&outputs[index])?
        };
        let mut extras = Vec::with_capacity(self.decoder_extras.len() + 1);
        for (name, value) in self.decoder_extras {
            extras.push((name.clone(), clone_value(value)?));
        }
        extras.push((binding.decoder_input.clone(), embeds));
        Ok(extras)
    }
}

impl DecodeLoopBackend for PipelineDecodeLoopBackend<'_> {
    fn context_len(&self) -> usize {
        self.context_tokens.len()
    }

    fn processor_prompt_tokens(&self) -> Vec<TokenId> {
        self.context_tokens.clone()
    }

    fn next_logits(&mut self) -> anyhow::Result<Vec<f32>> {
        let past_len = if self.decoder_state.use_kv {
            self.context_tokens
                .len()
                .saturating_sub(if self.generated_count == 0 {
                    self.prompt_len
                } else {
                    1
                })
        } else {
            0
        };
        let input_tokens = if self.decoder_state.use_kv && self.generated_count > 0 {
            self.context_tokens[self.context_tokens.len() - 1..].to_vec()
        } else {
            self.context_tokens.clone()
        };
        let extras = self.step_extras()?;
        let outputs = run_decode_step_with_extra(
            self.decoder,
            self.decoder_state,
            &input_tokens,
            past_len,
            &extras,
        )?;
        extract_next_token_logits(self.decoder, outputs)
    }

    fn commit_token(&mut self, token_id: TokenId) -> anyhow::Result<()> {
        self.context_tokens.push(token_id);
        self.generated_count += 1;
        Ok(())
    }
}

/// Executable plan for a pipeline, discriminated by strategy family.
///
/// Autoregressive pipelines drive a token decode loop (`generate`); single-pass
/// and iterative (diffusion) pipelines produce tensors (`run_pipeline`).
#[derive(Debug, Clone)]
enum PipelinePlan {
    Autoregressive(AutoregressivePlan),
    /// Dual, hierarchically-nested AR loops (multi-decoder TTS, DESIGN.md §20.3).
    NestedAutoregressive(NestedAutoregressivePlan),
    SinglePass(SinglePassPlan),
    Iterative(Box<IterativePlan>),
    /// Multi-stage pipeline (DESIGN.md §20): ordered stages run over a shared
    /// tensor pool, with dataflow edges routing tensors between them (e.g.
    /// audio-to-audio codec: encoder -> decoder; ASR/TTS encoders + vocoder).
    Composite(CompositePlan),
}

/// An ordered multi-stage pipeline. Each stage runs one or more component
/// sessions once, in declared order, over a shared tensor pool; the top-level
/// `dataflow` routes each stage's outputs into later stages' inputs.
#[derive(Debug, Clone)]
struct CompositePlan {
    stages: Vec<CompositeStage>,
    dataflow: Vec<DataflowEdge>,
}

/// One stage of a [`CompositePlan`].
#[derive(Debug, Clone)]
struct CompositeStage {
    /// Stage name (for diagnostics/timing), unique within the composite.
    name: String,
    kind: CompositeStageKind,
}

/// The execution strategy of a single composite stage.
#[derive(Debug, Clone)]
enum CompositeStageKind {
    /// Run one model once over the shared pool (encoder, codec, vocoder,
    /// embedder). Inputs are routed from the pool via the pipeline dataflow.
    SinglePass { model: String },
}

/// Token-by-token decoder pipeline (optionally with prompt-phase encoders and
/// post-decode single-pass stages).
///
/// The TTS shape (DESIGN.md §20) is `[encoders] -> AR decode -> vocoder`: the
/// decode loop emits audio *code* tokens, then one or more `final_only`
/// single-pass stages (a vocoder) run once over the shared pool to turn the
/// collected codes into a waveform. The generated code sequence is exposed to
/// those stages as the synthetic pool tensor `{decoder}.output_ids` of shape
/// `[1, num_generated]` (int64), routed to a stage input by a dataflow edge
/// (e.g. `decoder.output_ids -> vocoder.codes`).
#[derive(Debug, Clone)]
struct AutoregressivePlan {
    decoder: String,
    prompt_components: Vec<String>,
    /// Single-pass components declared `final_only`: run once, in declared
    /// order, after the decode loop completes (e.g. a TTS vocoder). Empty for a
    /// conventional text decoder or a Whisper-style ASR pipeline.
    post_decode_components: Vec<String>,
    dataflow: Vec<DataflowEdge>,
}

/// Dual, hierarchically-nested autoregressive pipeline — the multi-decoder TTS
/// (Qwen3-TTS-style) shape (DESIGN.md §20.3).
///
/// An **outer** AR loop (talker) runs up to `max_frames` frames; each outer step
/// is one audio frame and produces a per-frame `last_hidden_state`. That hidden
/// state seeds an **inner** AR loop (code_predictor) of `num_code_groups` steps
/// that emits the residual code groups for the frame. The inner loop threads the
/// seed at inner step 0 (via the dataflow edge
/// `{outer}.last_hidden_state -> {inner}.inputs_embeds`) and, on later steps, the
/// inner decoder's own per-code embedding output (threaded by the driver — no
/// dataflow self-edge, so the acyclic/single-producer validator stays happy).
///
/// All generated codes assemble into the synthetic pool tensor
/// `{outer}.output_codes` of shape `[1, frames, num_code_groups]` (int64), routed
/// to a post-decode `final_only` vocoder stage by a dataflow edge (e.g.
/// `talker.output_codes -> vocoder.codes`).
///
/// ## Pre-embedder mode (optional, backward compatible)
///
/// By default the outer talker is `input_ids`-driven (frame 0 = prompt tokens,
/// later frames = the talker's previous argmax token). When `pre_embedder` is
/// set, the talker is instead driven by `inputs_embeds` materialized each frame
/// from the PREVIOUS frame's codes `[outer_code_0, inner_code_1, ...,
/// inner_code_{num_code_groups-1}]` through a codec-sum pre-embedder component
/// (`frame_codes [+ text_embed] -> inputs_embeds`), matching the real Qwen3-TTS
/// talker. This keeps the engine generic: the codec-sum construction lives in an
/// ONNX component, not in Rust. See [`PreEmbedderBinding`]. The inner loop is
/// unchanged in both modes.
#[derive(Debug, Clone)]
struct NestedAutoregressivePlan {
    /// Outer decoder component (talker); one outer step == one audio frame.
    outer: String,
    /// Inner decoder component (code_predictor); expands one frame's residuals.
    inner: String,
    /// Inner-loop depth: code groups collected per frame (RVQ residual count).
    num_code_groups: usize,
    /// Maximum number of outer frames to generate.
    max_frames: usize,
    /// Outer decoder output port carrying the per-frame hidden state that seeds
    /// the inner loop (from the `{outer}.last_hidden_state -> {inner}.inputs_embeds`
    /// dataflow edge).
    outer_hidden_output: String,
    /// Inner decoder input port that receives the seed / threaded embedding.
    inner_embeds_input: String,
    /// Prompt-phase components (`prompt_only`), run once before the outer loop.
    prompt_components: Vec<String>,
    /// Post-decode components (`final_only`, e.g. a vocoder), run once after the
    /// outer loop over the shared pool (which holds `{outer}.output_codes`).
    post_decode_components: Vec<String>,
    /// Optional pre-embedder binding driving the outer talker via
    /// `inputs_embeds` (materialized codec-sum embedder) instead of `input_ids`.
    ///
    /// When `None` the outer loop is `input_ids`-driven and behaves exactly as
    /// before (backward compatible). When `Some`, each outer frame builds the
    /// talker's per-step `inputs_embeds` from the PREVIOUS frame's codes through
    /// the named pre-embedder component (see [`PreEmbedderBinding`]).
    pre_embedder: Option<PreEmbedderBinding>,
    dataflow: Vec<DataflowEdge>,
}

/// Wiring for a pre-embedder that drives the outer talker's per-step
/// `inputs_embeds` in a [`NestedAutoregressivePlan`].
///
/// The real Qwen3-TTS talker consumes `inputs_embeds` (not `input_ids`), built
/// each step from the previous frame's codes as `codec_sum(+ text_embed)`. On the
/// Mobius side that construction is materialized into an ONNX component with
/// inputs `frame_codes [batch, num_code_groups]` int64 (`[+ text_embed [batch, 1,
/// hidden]]`) → output `inputs_embeds [batch, 1, hidden]`. This binding records
/// the component name and the outer decoder input port fed by it; the exact
/// pre-embedder input names are resolved from its loaded session at drive time
/// (sessions are not available at plan-build time).
#[derive(Debug, Clone)]
struct PreEmbedderBinding {
    /// Pre-embedder component name (a declared model).
    component: String,
    /// Outer decoder input port that receives the per-step embeddings
    /// (`inputs_embeds`), from the required dataflow edge
    /// `{component}.inputs_embeds -> {outer}.inputs_embeds`.
    outer_input: String,
}

/// One forward invocation of a single model with no runtime-managed loop.
#[derive(Debug, Clone)]
struct SinglePassPlan {
    model: String,
    /// Components that run once before the model (e.g. an encoder).
    prompt_components: Vec<String>,
    dataflow: Vec<DataflowEdge>,
}

/// Coerce a float tensor to a model input's declared float dtype so the f32-space
/// pipeline math (schedulers, classifier-free guidance) can feed an fp16 / bf16
/// model and read its outputs back. Non-float dtypes and already-matching dtypes
/// are cloned unchanged.
fn coerce_value_to_dtype(value: &Value, target: DataType) -> anyhow::Result<Value> {
    if value.dtype() == target {
        return clone_value(value);
    }
    match (value.dtype(), target) {
        (
            DataType::Float32 | DataType::Float16 | DataType::BFloat16,
            DataType::Float32 | DataType::Float16 | DataType::BFloat16,
        ) => {
            let data = value.to_vec_f32_lossy()?;
            Value::from_f32_slice_as(&data, value.shape(), target)
                .map_err(|e| anyhow::anyhow!("failed to coerce value to {target:?}: {e}"))
        }
        _ => clone_value(value),
    }
}

/// Dump one iterative step's loop-carried tensor to `ONNX_GENAI_STEP_DUMP_DIR`
/// (when set) as `step_{i}_{port}.json` — used by the diffusion demo to animate
/// the reverse process. Best-effort; failures are ignored (never affects a run).
fn dump_iterative_step(denoiser: &str, port: &str, step: usize, value: &Value, step_ms: f64) {
    let Ok(dir) = std::env::var("ONNX_GENAI_STEP_DUMP_DIR") else {
        return;
    };
    let shape: Vec<i64> = value.shape().to_vec();
    // Emit int64 token sequences as integers (language diffusion) and everything
    // else as f32 (image latents). `step_ms` is this step's wall-clock time.
    let payload = match value.dtype() {
        DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8 => value
            .to_vec_i64()
            .ok()
            .map(|data| serde_json::json!({"dtype": "i64", "shape": shape, "data": data, "step_ms": step_ms})),
        _ => value
            .to_vec_f32()
            .ok()
            .map(|data| serde_json::json!({"dtype": "f32", "shape": shape, "data": data, "step_ms": step_ms})),
    };
    if let Some(payload) = payload {
        let path = std::path::Path::new(&dir).join(format!("step_{step:04}_{denoiser}_{port}.json"));
        let _ = std::fs::write(path, payload.to_string());
    }
}

/// Write the per-pipeline-stage timing report (`stages.json`) to the step-dump
/// directory when `ONNX_GENAI_STEP_DUMP_DIR` is set. Each entry is
/// `{component, phase, ms[, steps]}`, covering the prompt encoders (`encode`),
/// the denoiser loop total (`denoise`), and the final VAE-style pass (`decode`).
fn dump_stage_timings(stages: &[serde_json::Value]) {
    let Ok(dir) = std::env::var("ONNX_GENAI_STEP_DUMP_DIR") else {
        return;
    };
    let path = std::path::Path::new(&dir).join("stages.json");
    let _ = std::fs::write(path, serde_json::json!({ "stages": stages }).to_string());
}

#[derive(Debug, Clone)]
struct IterativePlan {
    /// The component re-invoked once per step.
    denoiser: String,
    /// Number of loop iterations.
    num_steps: usize,
    /// Classifier-free-guidance scale, carried for the scheduler follow-up.
    ///
    /// Not applied by this seam: CFG requires model-specific conditional /
    /// unconditional batching supplied by the scheduler registry (follow-up).
    guidance_scale: Option<f32>,
    /// Components run once before the loop (e.g. a text/prompt encoder).
    prompt_components: Vec<String>,
    /// Components run once after the loop (`final_only`, e.g. a VAE decoder).
    final_components: Vec<String>,
    /// Loop-carried edges internal to the denoiser: `(output_port, input_port)`.
    ///
    /// Each step i>0 feeds step (i-1)'s `output_port` into `input_port`. Step 0
    /// reads the seed from the external `denoiser.input_port` tensor.
    loop_edges: Vec<(String, String)>,
    /// Denoiser input port that receives the per-step timestep scalar, if any.
    timestep_input: Option<String>,
    /// First step index (0 for txt2img; >0 for a partial img2img denoise loop).
    start_step: usize,
    /// Explicit per-step timestep schedule (length == `num_steps`); when absent
    /// the 0-based step index is fed instead.
    timesteps: Option<Vec<f32>>,
    /// Optional scheduler applied to loop-carried edges (`None` = identity
    /// feedback). Built from the registry by `scheduler_config.kind`.
    scheduler: Option<Arc<dyn Scheduler>>,
    /// CFG conditioning input port zeroed on the unconditional pass (set only
    /// when guidance is active).
    cfg_conditioning_input: Option<String>,
    dataflow: Vec<DataflowEdge>,
    /// The declared scheduler config, kept so a per-request `num_steps` override
    /// can rebuild the scheduler (whose schedule may be baked at build time).
    scheduler_spec: Option<SchedulerSpec>,
    /// The scheduler registry, kept for the same per-request rebuild.
    scheduler_registry: SchedulerRegistry,
}

/// A loop-carried transform applied to a denoiser's output at each iterative
/// step. **Implement this trait to plug in a custom scheduler** and register it
/// with a [`SchedulerRegistry`]; the built-in `ddim` (continuous latents) and
/// `masked_diffusion` (discrete tokens) are just implementations.
///
/// `sample` is the value currently fed to the loop-carried input; `model_output`
/// is the denoiser's output this step. Return the next loop-carried value.
pub trait Scheduler: Send + Sync + std::fmt::Debug {
    fn step(
        &self,
        step: usize,
        num_steps: usize,
        sample: &Value,
        model_output: &Value,
    ) -> anyhow::Result<Value>;

    /// Reset any per-loop internal state (e.g. a multistep scheduler's previous
    /// prediction). Called once before each denoise loop. Default no-op.
    fn reset(&self) {}

    /// Whether this scheduler consumes fresh Gaussian noise each step (ancestral /
    /// stochastic samplers). When `true`, the loop supplies per-step noise via
    /// [`Scheduler::step_with_noise`]. Default `false` (deterministic).
    fn needs_noise(&self) -> bool {
        false
    }

    /// Like [`Scheduler::step`] but with the per-step noise an ancestral sampler
    /// needs. The default ignores `noise` and delegates to `step`, so existing
    /// deterministic schedulers are unaffected.
    fn step_with_noise(
        &self,
        step: usize,
        num_steps: usize,
        sample: &Value,
        model_output: &Value,
        _noise: Option<&Value>,
    ) -> anyhow::Result<Value> {
        self.step(step, num_steps, sample, model_output)
    }

    /// Per-step transform applied to the loop-carried input BEFORE the denoiser
    /// (e.g. Euler's `sample / sqrt(sigma^2 + 1)`). `Ok(None)` = identity (the
    /// denoiser sees the raw loop-carried value, as DDIM requires).
    fn scale_input(
        &self,
        _step: usize,
        _num_steps: usize,
        _sample: &Value,
    ) -> anyhow::Result<Option<Value>> {
        Ok(None)
    }

    /// The factor by which the caller must scale the initial random latent so it
    /// lives in this scheduler's sigma space. Sigma-space samplers (Euler,
    /// Euler-Ancestral) return their maximum sigma (`sigmas[0]`); DDIM and
    /// DPM-Solver++ leave the seed unscaled and return `1.0` (the default).
    fn init_noise_sigma(&self) -> f32 {
        1.0
    }

    /// The per-step denoiser timesteps this scheduler feeds to the model, matching
    /// the diffusers scheduler it emulates (e.g. `[999.0, 966.0, ..., 33.0]` for a
    /// 30-step DPM-Solver++ linspace schedule). Length equals the loop step count.
    ///
    /// The pipeline uses these when the plan does not carry an explicit
    /// `strategy.timesteps` schedule, so from-scratch packages that omit the
    /// timestep table still drive the denoiser with the correct diffusion
    /// timesteps rather than the raw `0..num_steps` step index. Returns `None`
    /// for schedulers with no meaningful timestep (e.g. discrete token diffusion),
    /// leaving the pipeline to fall back to the step index.
    fn timesteps(&self) -> Option<Vec<f32>> {
        None
    }

    /// Build the unconditional loop-carried sample for classifier-free guidance
    /// from the current (conditional) one, when the guidance direction is a
    /// transform of the loop state rather than a separate conditioning input.
    ///
    /// Discrete language diffusion (LLaDA) forms its unconditional pass by
    /// re-masking the prompt tokens of the current sequence (`un_x[prompt] =
    /// mask_id`); the pipeline feeds the returned value as the denoiser's
    /// loop-carried input on the unconditional pass. Continuous (image)
    /// schedulers return `None` (their unconditional direction comes from a
    /// zeroed / `.uncond` conditioning input instead).
    fn cfg_uncond_sample(&self, _sample: &Value) -> anyhow::Result<Option<Value>> {
        Ok(None)
    }
}

/// Builds a [`Scheduler`] from a declared [`SchedulerSpec`] and the loop length.
pub type SchedulerFactory =
    Arc<dyn Fn(&SchedulerSpec, usize) -> anyhow::Result<Arc<dyn Scheduler>> + Send + Sync>;

/// Registry mapping a `scheduler_config.kind` string to a factory. Users extend
/// it with [`register`](Self::register) to support their own schedulers, then
/// load a pipeline via [`PipelineEngine::from_pipeline_dir_with_schedulers`].
#[derive(Clone)]
pub struct SchedulerRegistry {
    factories: HashMap<String, SchedulerFactory>,
}

impl std::fmt::Debug for SchedulerRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SchedulerRegistry")
            .field("kinds", &self.factories.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl SchedulerRegistry {
    /// Registry with the built-in `ddim`, `euler`, `dpmpp_2m` and `masked_diffusion` schedulers.
    pub fn builtin() -> Self {
        let mut factories: HashMap<String, SchedulerFactory> = HashMap::new();
        factories.insert(
            "ddim".to_string(),
            Arc::new(|cfg: &SchedulerSpec, num_steps: usize| {
                if let Some(prediction) = cfg.prediction_type.as_deref()
                    && prediction != "epsilon"
                {
                    anyhow::bail!(
                        "unsupported ddim prediction_type '{prediction}' (only 'epsilon')"
                    );
                }
                let sched = DdimSchedule::with_schedule(
                    cfg.num_train_timesteps.unwrap_or(1000),
                    cfg.beta_start.unwrap_or(0.00085),
                    cfg.beta_end.unwrap_or(0.012),
                    cfg.beta_schedule.as_deref().unwrap_or("linear"),
                    num_steps,
                )?;
                Ok(Arc::new(sched) as Arc<dyn Scheduler>)
            }),
        );
        factories.insert(
            "euler".to_string(),
            Arc::new(|cfg: &SchedulerSpec, num_steps: usize| {
                if let Some(prediction) = cfg.prediction_type.as_deref()
                    && prediction != "epsilon"
                {
                    anyhow::bail!(
                        "unsupported euler prediction_type '{prediction}' (only 'epsilon')"
                    );
                }
                let sched = EulerSchedule::with_schedule(
                    cfg.num_train_timesteps.unwrap_or(1000),
                    cfg.beta_start.unwrap_or(0.00085),
                    cfg.beta_end.unwrap_or(0.012),
                    cfg.beta_schedule.as_deref().unwrap_or("scaled_linear"),
                    num_steps,
                    sigma_spacing(cfg)?,
                )?;
                Ok(Arc::new(sched) as Arc<dyn Scheduler>)
            }),
        );
        factories.insert(
            "euler_ancestral".to_string(),
            Arc::new(|cfg: &SchedulerSpec, num_steps: usize| {
                if let Some(prediction) = cfg.prediction_type.as_deref()
                    && prediction != "epsilon"
                {
                    anyhow::bail!(
                        "unsupported euler_ancestral prediction_type '{prediction}' (only 'epsilon')"
                    );
                }
                let sched = EulerAncestral::with_schedule(
                    cfg.num_train_timesteps.unwrap_or(1000),
                    cfg.beta_start.unwrap_or(0.00085),
                    cfg.beta_end.unwrap_or(0.012),
                    cfg.beta_schedule.as_deref().unwrap_or("scaled_linear"),
                    num_steps,
                    sigma_spacing(cfg)?,
                )?;
                Ok(Arc::new(sched) as Arc<dyn Scheduler>)
            }),
        );
        factories.insert(
            "dpmpp_2m".to_string(),
            Arc::new(|cfg: &SchedulerSpec, num_steps: usize| {
                if let Some(prediction) = cfg.prediction_type.as_deref()
                    && prediction != "epsilon"
                {
                    anyhow::bail!(
                        "unsupported dpmpp_2m prediction_type '{prediction}' (only 'epsilon')"
                    );
                }
                let sched = Dpmpp2m::with_schedule(
                    cfg.num_train_timesteps.unwrap_or(1000),
                    cfg.beta_start.unwrap_or(0.00085),
                    cfg.beta_end.unwrap_or(0.012),
                    cfg.beta_schedule.as_deref().unwrap_or("scaled_linear"),
                    num_steps,
                    sigma_spacing(cfg)?,
                )?;
                Ok(Arc::new(sched) as Arc<dyn Scheduler>)
            }),
        );
        factories.insert(
            "masked_diffusion".to_string(),
            Arc::new(|cfg: &SchedulerSpec, _num_steps: usize| {
                let mask_token_id = cfg
                    .mask_token_id
                    .context("masked_diffusion scheduler requires 'mask_token_id'")?;
                let temperature = cfg.temperature.unwrap_or(0.0);
                if temperature < 0.0 {
                    anyhow::bail!("masked_diffusion temperature must be >= 0");
                }
                if let Some(block_length) = cfg.block_length
                    && block_length == 0
                {
                    anyhow::bail!("masked_diffusion block_length must be >= 1");
                }
                let remasking = match cfg.remasking.as_deref() {
                    None | Some("low_confidence") => Remasking::LowConfidence,
                    Some("random") => Remasking::Random,
                    Some(other) => anyhow::bail!(
                        "masked_diffusion remasking must be 'low_confidence' or 'random', \
                         got '{other}'"
                    ),
                };
                Ok(Arc::new(MaskedDiffusion {
                    mask_token_id,
                    temperature,
                    block_length: cfg.block_length,
                    remasking,
                    generation_start: Mutex::new(None),
                }) as Arc<dyn Scheduler>)
            }),
        );
        Self { factories }
    }

    /// Register (or override) a scheduler kind with a factory.
    pub fn register(&mut self, kind: impl Into<String>, factory: SchedulerFactory) {
        self.factories.insert(kind.into(), factory);
    }

    fn build(&self, spec: &SchedulerSpec, num_steps: usize) -> anyhow::Result<Arc<dyn Scheduler>> {
        let factory = self.factories.get(&spec.kind).with_context(|| {
            format!(
                "unknown scheduler kind '{}' (registered: {:?})",
                spec.kind,
                self.factories.keys().collect::<Vec<_>>()
            )
        })?;
        factory(spec, num_steps)
    }
}

impl Default for SchedulerRegistry {
    fn default() -> Self {
        Self::builtin()
    }
}

/// Masked (discrete) language diffusion with two unmasking strategies, selected
/// by the scheduler config's `remasking` field.
///
/// The loop-carried tensor is an int64 token sequence `[B, S]` (prompt tokens
/// plus a masked generation region), the denoiser emits `[B, S, V]` logits, and
/// each step unmasks a growing subset of the still-masked positions until all
/// are filled by the final step.
///
/// **`remasking = "low_confidence"` (default)** — faithful to `ML-GSAI/LLaDA`'s
/// `generate` (`cfg_scale = 0`):
///   * the chosen token per position is the argmax of the (optionally
///     Gumbel-noised) logits (`add_gumbel_noise`; identity at `temperature = 0`);
///   * the confidence that ranks positions for remasking is the clean-softmax
///     probability of that chosen token (`remasking = "low_confidence"`);
///   * each step commits the highest-confidence still-masked positions, split
///     evenly across steps (`ceil(remaining / remaining_steps)`).
///
/// **`remasking = "random"`** — MDLM-style ancestral sampling (Sahoo et al.):
///   * each still-masked position unmasks *independently* with the schedule
///     probability `1/(steps_remaining_in_block)` (so the expected unmasked
///     fraction matches the log-linear absorbing schedule, and the final step
///     unmasks everything);
///   * on unmasking, the token is *sampled* from the model's categorical
///     distribution via the Gumbel-max trick (`temperature = 1.0` is a true
///     categorical sample; `0` is greedy argmax). The mask token id is never
///     emitted (SUBS parameterization). This per-position stochastic unmasking
///     avoids the degenerate repetition confidence-ranked greedy decoding
///     produces on non-LLaDA checkpoints such as MDLM.
///
/// With `block_length` set, the generation region is split into contiguous
/// left-to-right blocks; the total `num_steps` is divided evenly across the
/// `num_blocks`, and each step only unmasks tokens inside the current block
/// (semi-autoregressive remasking). A single block (the default) spans the
/// whole masked region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Remasking {
    /// LLaDA confidence-ranked commit (default).
    LowConfidence,
    /// MDLM-style per-position stochastic ancestral unmasking.
    Random,
}

#[derive(Debug)]
struct MaskedDiffusion {
    mask_token_id: i64,
    temperature: f32,
    block_length: Option<usize>,
    /// Unmasking strategy (see [`Remasking`]).
    remasking: Remasking,
    /// Per-sequence generation-region start (prompt length), captured on the
    /// first step of a loop and cleared by [`Scheduler::reset`]. This lets the
    /// semi-autoregressive block boundaries be derived without threading the
    /// prompt length through the [`Scheduler`] trait.
    generation_start: Mutex<Option<Vec<usize>>>,
}

impl MaskedDiffusion {
    /// Capture each sequence's generation-region start (prompt length) on the
    /// first use of a loop — the index of its first mask token. Cleared by
    /// [`Scheduler::reset`]. Called from both `step` and `cfg_uncond_sample`, so
    /// whichever runs first in a loop iteration records it from the seed.
    fn ensure_generation_start(&self, tokens: &[i64], batch: usize, sequence_length: usize) {
        let mut guard = self.generation_start.lock().unwrap();
        if guard.is_some() {
            return;
        }
        let mut starts = Vec::with_capacity(batch);
        for row_index in 0..batch {
            let start = row_index * sequence_length;
            let first_mask = tokens[start..start + sequence_length]
                .iter()
                .position(|&token| token == self.mask_token_id)
                .unwrap_or(sequence_length);
            starts.push(first_mask);
        }
        *guard = Some(starts);
    }

    /// Argmax token id and its clean-softmax confidence for one logit row.
    ///
    /// `gumbel` supplies one uniform sample in `(0, 1)` per vocab entry when
    /// `temperature > 0`; it is ignored (and may be empty) at `temperature = 0`.
    fn predict_row(&self, row: &[f32], gumbel: &[f64]) -> (i64, f32) {
        // Clean-softmax denominator (numerically stable) for the confidence.
        let max_logit = row.iter().copied().fold(f32::MIN, f32::max);
        let sum_exp: f32 = row.iter().map(|&x| (x - max_logit).exp()).sum();

        // Chosen token: argmax of the (optionally Gumbel-noised) logits.
        // LLaDA: logits.exp() / (-log u)^temperature, i.e. argmax of
        // `logit - temperature * ln(-ln u)`.
        let mut best_index = 0usize;
        let mut best_score = f32::MIN;
        for (j, &logit) in row.iter().enumerate() {
            let score = if self.temperature > 0.0 {
                let u = gumbel[j];
                logit - self.temperature * (-u.ln()).ln() as f32
            } else {
                logit
            };
            if score > best_score {
                best_score = score;
                best_index = j;
            }
        }
        let confidence = (row[best_index] - max_logit).exp() / sum_exp;
        (best_index as i64, confidence)
    }

    /// Sample one token from a logit row for MDLM-style ancestral unmasking.
    ///
    /// Uses the Gumbel-max trick so `temperature = 1.0` draws a true categorical
    /// sample from `softmax(logits)`, while `temperature = 0` is greedy argmax
    /// (matching [`predict_row`]'s token choice). The mask token id is excluded
    /// (SUBS parameterization: an unmasked position is never re-set to mask).
    fn sample_token(&self, row: &[f32], step: usize, position: usize, vocab: usize) -> i64 {
        let gumbel = if self.temperature > 0.0 {
            gumbel_uniforms(step, position, vocab)
        } else {
            Vec::new()
        };
        let mut best_index: Option<usize> = None;
        let mut best_score = f32::MIN;
        for (j, &logit) in row.iter().enumerate() {
            if j as i64 == self.mask_token_id {
                continue;
            }
            let score = if self.temperature > 0.0 {
                logit - self.temperature * (-gumbel[j].ln()).ln() as f32
            } else {
                logit
            };
            if best_index.is_none() || score > best_score {
                best_score = score;
                best_index = Some(j);
            }
        }
        best_index.unwrap_or(0) as i64
    }
}

impl Scheduler for MaskedDiffusion {
    fn reset(&self) {
        *self.generation_start.lock().unwrap() = None;
    }

    /// LLaDA unconditional pass: re-mask the prompt tokens of the current
    /// sequence (`un_x[prompt] = mask_id`), leaving the generation region as-is.
    fn cfg_uncond_sample(&self, sample: &Value) -> anyhow::Result<Option<Value>> {
        let shape = sample.shape().to_vec();
        let tokens = sample.to_vec_i64()?;
        let count = tokens.len();
        let sequence_length = *shape.last().unwrap_or(&(count as i64)) as usize;
        if sequence_length == 0 {
            return Ok(None);
        }
        let batch = count.checked_div(sequence_length).unwrap_or(0).max(1);
        self.ensure_generation_start(&tokens, batch, sequence_length);
        let generation_start = self.generation_start.lock().unwrap().clone().unwrap();

        let mut output = tokens;
        for (row_index, &prompt_length) in generation_start.iter().enumerate() {
            let row_start = row_index * sequence_length;
            for offset in 0..prompt_length.min(sequence_length) {
                output[row_start + offset] = self.mask_token_id;
            }
        }
        Value::from_slice_i64(&output, &shape).map(Some).map_err(Into::into)
    }

    fn step(
        &self,
        step: usize,
        num_steps: usize,
        tokens: &Value,
        logits: &Value,
    ) -> anyhow::Result<Value> {
        let token_shape = tokens.shape().to_vec();
        let tokens = tokens.to_vec_i64()?;
        let sequence_count = tokens.len();
        let logit_shape = logits.shape();
        let vocab = *logit_shape
            .last()
            .context("masked_diffusion logits must be rank >= 1")? as usize;
        if vocab == 0 || sequence_count == 0 || logits.numel() != sequence_count * vocab {
            anyhow::bail!(
                "masked_diffusion shape mismatch: tokens {token_shape:?}, logits {logit_shape:?}"
            );
        }
        // Split the flat token buffer into per-sequence rows so top-k selection
        // and the transfer schedule are computed independently per sequence
        // (matching LLaDA's per-batch-row `topk`). Rank-1 inputs are one row.
        let sequence_length = *token_shape.last().unwrap_or(&(sequence_count as i64)) as usize;
        let batch = sequence_count.checked_div(sequence_length).unwrap_or(0).max(1);

        // Capture each sequence's generation-region start on the first step.
        self.ensure_generation_start(&tokens, batch, sequence_length);
        let generation_start = self.generation_start.lock().unwrap().clone().unwrap();

        let all_logits = logits.to_vec_f32()?;
        let mut output = tokens.clone();

        for (row_index, &prompt_length) in generation_start.iter().enumerate() {
            let row_start = row_index * sequence_length;
            let generation_length = sequence_length.saturating_sub(prompt_length);
            if generation_length == 0 {
                continue;
            }
            let block_length = self
                .block_length
                .unwrap_or(generation_length)
                .min(generation_length)
                .max(1);
            if !generation_length.is_multiple_of(block_length) {
                anyhow::bail!(
                    "masked_diffusion: generation length {generation_length} is not divisible \
                     by block_length {block_length}"
                );
            }
            let num_blocks = generation_length / block_length;
            if !num_steps.is_multiple_of(num_blocks) {
                anyhow::bail!(
                    "masked_diffusion: num_steps {num_steps} is not divisible by num_blocks \
                     {num_blocks} (generation_length {generation_length} / block_length \
                     {block_length})"
                );
            }
            let steps_per_block = num_steps / num_blocks;
            let block_index = (step / steps_per_block).min(num_blocks - 1);
            let step_in_block = step % steps_per_block;
            let block_start = prompt_length + block_index * block_length;
            let block_end = (block_start + block_length).min(sequence_length);
            let remaining_steps_in_block = steps_per_block - step_in_block;

            match self.remasking {
                Remasking::LowConfidence => {
                    // Predicted token + confidence for every still-masked position
                    // inside the current block (only these are candidates).
                    let mut candidates: Vec<(usize, i64, f32)> = Vec::new();
                    for offset in block_start..block_end {
                        let position = row_start + offset;
                        if tokens[position] != self.mask_token_id {
                            continue;
                        }
                        let logit_row = &all_logits[position * vocab..(position + 1) * vocab];
                        let gumbel = if self.temperature > 0.0 {
                            gumbel_uniforms(step, position, vocab)
                        } else {
                            Vec::new()
                        };
                        let (predicted, confidence) = self.predict_row(logit_row, &gumbel);
                        candidates.push((position, predicted, confidence));
                    }
                    if candidates.is_empty() {
                        continue;
                    }
                    // Commit the highest-confidence subset for this block-step. The
                    // even split of the block's masked count across its remaining
                    // steps equals ceil(remaining / remaining_steps_in_block).
                    candidates
                        .sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
                    let commit = candidates.len().div_ceil(remaining_steps_in_block);
                    for &(position, predicted, _) in candidates.iter().take(commit) {
                        output[position] = predicted;
                    }
                }
                Remasking::Random => {
                    // MDLM ancestral update: each still-masked position in the block
                    // unmasks independently with probability 1/(steps remaining), so
                    // the expected unmasked fraction follows the log-linear schedule
                    // and the block's final step unmasks everything. The token is
                    // sampled from the model's categorical distribution.
                    let last_step_in_block = remaining_steps_in_block <= 1;
                    let unmask_prob = 1.0f64 / remaining_steps_in_block as f64;
                    for offset in block_start..block_end {
                        let position = row_start + offset;
                        if tokens[position] != self.mask_token_id {
                            continue;
                        }
                        if last_step_in_block || unmask_uniform(step, position) < unmask_prob {
                            let logit_row =
                                &all_logits[position * vocab..(position + 1) * vocab];
                            output[position] = self.sample_token(logit_row, step, position, vocab);
                        }
                    }
                }
            }
        }
        Value::from_slice_i64(&output, &token_shape).map_err(Into::into)
    }
}

/// One uniform sample in `(0, 1)` per vocab entry for Gumbel-max sampling,
/// seeded deterministically from `(step, position)` so a run is reproducible.
///
/// Note: this is reproducible across onnx-genai runs but is NOT bit-identical to
/// LLaDA's `torch.rand`-based sampling; parity tests exercise `temperature = 0`.
fn gumbel_uniforms(step: usize, position: usize, vocab: usize) -> Vec<f64> {
    use rand::{Rng, SeedableRng};
    let seed = (step as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(position as u64);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..vocab)
        // Clamp away from 0 and 1 to keep -ln(-ln u) finite.
        .map(|_| rng.random::<f64>().clamp(1e-9, 1.0 - 1e-9))
        .collect()
}

/// One uniform sample in `[0, 1)` per `(step, position)` for the MDLM ancestral
/// per-position unmask decision, seeded deterministically (so a run is
/// reproducible) but with a distinct mix from [`gumbel_uniforms`] so the unmask
/// decision is independent of the token-sampling noise at the same position.
fn unmask_uniform(step: usize, position: usize) -> f64 {
    use rand::{Rng, SeedableRng};
    let seed = (step as u64)
        .wrapping_mul(0x2545_F491_4F6C_DD1D)
        .wrapping_add((position as u64).wrapping_mul(0xD1B5_4A32_D192_ED03))
        .wrapping_add(0xA0761_D6478_BD642F);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    rng.random::<f64>()
}

/// DDIM (η = 0, epsilon-prediction) noise schedule, precomputed per inference
/// step as `(alpha_cumprod_t, alpha_cumprod_prev)`.
///
/// Diffusion-standard update for a model that predicts noise `eps`:
///   `x0_hat = (x_t - sqrt(1 - a_t) * eps) / sqrt(a_t)`
///   `x_prev = sqrt(a_prev) * x0_hat + sqrt(1 - a_prev) * eps`
#[derive(Debug, Clone)]
struct DdimSchedule {
    steps: Vec<(f32, f32)>,
    timesteps: Vec<f32>,
}

impl DdimSchedule {
    fn with_schedule(
        num_train_timesteps: usize,
        beta_start: f32,
        beta_end: f32,
        beta_schedule: &str,
        num_steps: usize,
    ) -> anyhow::Result<Self> {
        if num_train_timesteps < 2 {
            anyhow::bail!("scheduler num_train_timesteps must be >= 2");
        }
        if num_steps == 0 || num_steps > num_train_timesteps {
            anyhow::bail!(
                "scheduler num_steps ({num_steps}) must be in 1..={num_train_timesteps}"
            );
        }
        // Beta schedule -> cumulative product of alphas.
        //   linear:        beta_i = lerp(beta_start, beta_end)
        //   scaled_linear: beta_i = lerp(sqrt(beta_start), sqrt(beta_end))^2  (Stable Diffusion)
        let denom = (num_train_timesteps - 1) as f32;
        let (lo, hi, square) = match beta_schedule {
            "linear" => (beta_start, beta_end, false),
            "scaled_linear" => (beta_start.sqrt(), beta_end.sqrt(), true),
            other => anyhow::bail!(
                "unsupported scheduler beta_schedule '{other}' (expected 'linear' or 'scaled_linear')"
            ),
        };
        let mut alpha_cumprod = Vec::with_capacity(num_train_timesteps);
        let mut prod = 1.0f32;
        for i in 0..num_train_timesteps {
            let mut beta = lo + (hi - lo) * (i as f32) / denom;
            if square {
                beta *= beta;
            }
            prod *= 1.0 - beta;
            alpha_cumprod.push(prod);
        }
        // Evenly spaced inference timesteps, descending (diffusers convention).
        let step_ratio = num_train_timesteps / num_steps;
        let ascending: Vec<usize> = (0..num_steps).map(|i| i * step_ratio).collect();
        let mut steps = Vec::with_capacity(num_steps);
        let mut timesteps = Vec::with_capacity(num_steps);
        for k in 0..num_steps {
            let t = ascending[num_steps - 1 - k];
            timesteps.push(t as f32);
            let a_t = alpha_cumprod[t];
            let a_prev = if k + 1 < num_steps {
                alpha_cumprod[ascending[num_steps - 1 - (k + 1)]]
            } else {
                1.0
            };
            steps.push((a_t, a_prev));
        }
        Ok(Self { steps, timesteps })
    }

    /// Apply one DDIM step to `sample` given the model's noise prediction `eps`.
    fn step(&self, k: usize, sample: &[f32], eps: &[f32]) -> anyhow::Result<Vec<f32>> {
        if sample.len() != eps.len() {
            anyhow::bail!(
                "scheduler sample/eps length mismatch: {} vs {}",
                sample.len(),
                eps.len()
            );
        }
        let (a_t, a_prev) = self.steps[k];
        let sqrt_a_t = a_t.sqrt();
        let sqrt_one_minus_a_t = (1.0 - a_t).sqrt();
        let sqrt_a_prev = a_prev.sqrt();
        let sqrt_one_minus_a_prev = (1.0 - a_prev).sqrt();
        Ok(sample
            .iter()
            .zip(eps)
            .map(|(&x, &e)| {
                let x0_hat = (x - sqrt_one_minus_a_t * e) / sqrt_a_t;
                sqrt_a_prev * x0_hat + sqrt_one_minus_a_prev * e
            })
            .collect())
    }
}

impl Scheduler for DdimSchedule {
    fn step(
        &self,
        step: usize,
        _num_steps: usize,
        sample: &Value,
        model_output: &Value,
    ) -> anyhow::Result<Value> {
        let shape = sample.shape().to_vec();
        let stepped =
            DdimSchedule::step(self, step, &sample.to_vec_f32_lossy()?, &model_output.to_vec_f32_lossy()?)?;
        Value::from_slice_f32(&stepped, &shape).map_err(Into::into)
    }

    fn timesteps(&self) -> Option<Vec<f32>> {
        Some(self.timesteps.clone())
    }
}

/// Euler (`EulerDiscreteScheduler`, epsilon prediction) — a sigma-space
/// scheduler. Unlike DDIM it rescales the loop-carried sample before the
/// denoiser (`x / sqrt(sigma^2 + 1)`), then advances the *raw* sample along the
/// noise derivative: `x_next = x + eps * (sigma_next - sigma)`. Matches diffusers
/// `EulerDiscreteScheduler(timestep_spacing="linspace", interpolation_type="linear")`.
/// The initial seed must be pre-scaled by `init_noise_sigma` (= `sigmas[0]`).
#[derive(Debug, Clone)]
struct EulerSchedule {
    /// Inference sigmas, descending, with a trailing `0.0`. Length `num_steps + 1`.
    sigmas: Vec<f32>,
    /// Per-step denoiser timesteps (fractional), length `num_steps`.
    timesteps: Vec<f32>,
}

impl EulerSchedule {
    fn with_schedule(
        num_train_timesteps: usize,
        beta_start: f32,
        beta_end: f32,
        beta_schedule: &str,
        num_steps: usize,
        spacing: &str,
    ) -> anyhow::Result<Self> {
        if num_train_timesteps < 2 {
            anyhow::bail!("scheduler num_train_timesteps must be >= 2");
        }
        if num_steps == 0 || num_steps > num_train_timesteps {
            anyhow::bail!(
                "scheduler num_steps ({num_steps}) must be in 1..={num_train_timesteps}"
            );
        }
        if let Some(sigmas) =
            spacing_sigmas(spacing, num_train_timesteps, beta_start, beta_end, beta_schedule, num_steps)?
        {
            let train = training_sigmas(num_train_timesteps, beta_start, beta_end, beta_schedule)?;
            let timesteps = sigmas[..num_steps].iter().map(|&s| sigma_to_t(&train, s)).collect();
            return Ok(Self { sigmas, timesteps });
        }
        let denom = (num_train_timesteps - 1) as f32;
        let (lo, hi, square) = match beta_schedule {
            "linear" => (beta_start, beta_end, false),
            "scaled_linear" => (beta_start.sqrt(), beta_end.sqrt(), true),
            other => anyhow::bail!(
                "unsupported scheduler beta_schedule '{other}' (expected 'linear' or 'scaled_linear')"
            ),
        };
        // Training sigmas: sigma_i = sqrt((1 - alpha_cumprod_i) / alpha_cumprod_i).
        let mut train_sigmas = Vec::with_capacity(num_train_timesteps);
        let mut prod = 1.0f32;
        for i in 0..num_train_timesteps {
            let mut beta = lo + (hi - lo) * (i as f32) / denom;
            if square {
                beta *= beta;
            }
            prod *= 1.0 - beta;
            train_sigmas.push(((1.0 - prod) / prod).sqrt());
        }
        // "linspace" timesteps: evenly spaced over [0, N-1], taken descending,
        // with sigmas linearly interpolated at each (fractional) timestep.
        let ts_denom = if num_steps > 1 { (num_steps - 1) as f32 } else { 1.0 };
        let interp = |t: f32| -> f32 {
            let low = t.floor().max(0.0) as usize;
            let high = (low + 1).min(num_train_timesteps - 1);
            let frac = t - low as f32;
            train_sigmas[low] * (1.0 - frac) + train_sigmas[high] * frac
        };
        let mut sigmas = Vec::with_capacity(num_steps + 1);
        let mut timesteps = Vec::with_capacity(num_steps);
        for k in 0..num_steps {
            let idx = num_steps - 1 - k;
            let t = idx as f32 * denom / ts_denom;
            timesteps.push(t);
            sigmas.push(interp(t));
        }
        sigmas.push(0.0);
        Ok(Self { sigmas, timesteps })
    }

    /// `x / sqrt(sigma^2 + 1)` — scale the raw sample for the denoiser input.
    fn scale(&self, step: usize, sample: &[f32]) -> Vec<f32> {
        let factor = (self.sigmas[step] * self.sigmas[step] + 1.0).sqrt();
        sample.iter().map(|&x| x / factor).collect()
    }

    /// `x_next = x + eps * (sigma_next - sigma)` on the raw sample.
    fn step_vec(&self, step: usize, sample: &[f32], eps: &[f32]) -> anyhow::Result<Vec<f32>> {
        if sample.len() != eps.len() {
            anyhow::bail!(
                "scheduler sample/eps length mismatch: {} vs {}",
                sample.len(),
                eps.len()
            );
        }
        let dt = self.sigmas[step + 1] - self.sigmas[step];
        Ok(sample.iter().zip(eps).map(|(&x, &e)| x + e * dt).collect())
    }
}

impl Scheduler for EulerSchedule {
    fn step(
        &self,
        step: usize,
        _num_steps: usize,
        sample: &Value,
        model_output: &Value,
    ) -> anyhow::Result<Value> {
        let shape = sample.shape().to_vec();
        let stepped = self.step_vec(
            step,
            &sample.to_vec_f32_lossy()?,
            &model_output.to_vec_f32_lossy()?,
        )?;
        Value::from_slice_f32(&stepped, &shape).map_err(Into::into)
    }

    fn scale_input(
        &self,
        step: usize,
        _num_steps: usize,
        sample: &Value,
    ) -> anyhow::Result<Option<Value>> {
        let shape = sample.shape().to_vec();
        let scaled = self.scale(step, &sample.to_vec_f32_lossy()?);
        Ok(Some(Value::from_slice_f32(&scaled, &shape)?))
    }

    fn init_noise_sigma(&self) -> f32 {
        self.sigmas[0]
    }

    fn timesteps(&self) -> Option<Vec<f32>> {
        Some(self.timesteps.clone())
    }
}

/// Euler Ancestral (`EulerAncestralDiscreteScheduler`, epsilon) — a *stochastic*
/// sampler (one of the most-used in ComfyUI). Like Euler it scales the model
/// input and seeds at `sigmas[0]`, but each step advances to an intermediate
/// `sigma_down` and injects fresh noise scaled by `sigma_up`:
///   `sigma_up   = sqrt(sigma_to^2 (sigma_from^2 - sigma_to^2) / sigma_from^2)`
///   `sigma_down = sqrt(sigma_to^2 - sigma_up^2)`
///   `x_next = x + eps*(sigma_down - sigma) + noise*sigma_up`.
/// Matches diffusers when fed the same per-step noise sequence.
#[derive(Debug, Clone)]
struct EulerAncestral {
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
}

impl EulerAncestral {
    fn with_schedule(
        num_train_timesteps: usize,
        beta_start: f32,
        beta_end: f32,
        beta_schedule: &str,
        num_steps: usize,
        spacing: &str,
    ) -> anyhow::Result<Self> {
        // Same sigma schedule as Euler (linspace interp / Karras / exponential).
        let euler = EulerSchedule::with_schedule(
            num_train_timesteps,
            beta_start,
            beta_end,
            beta_schedule,
            num_steps,
            spacing,
        )?;
        Ok(Self { sigmas: euler.sigmas, timesteps: euler.timesteps })
    }
}

impl Scheduler for EulerAncestral {
    fn step(
        &self,
        _step: usize,
        _num_steps: usize,
        _sample: &Value,
        _model_output: &Value,
    ) -> anyhow::Result<Value> {
        anyhow::bail!("euler_ancestral is stochastic; the loop must call step_with_noise")
    }

    fn needs_noise(&self) -> bool {
        true
    }

    fn step_with_noise(
        &self,
        step: usize,
        _num_steps: usize,
        sample: &Value,
        model_output: &Value,
        noise: Option<&Value>,
    ) -> anyhow::Result<Value> {
        let shape = sample.shape().to_vec();
        let x = sample.to_vec_f32_lossy()?;
        let eps = model_output.to_vec_f32_lossy()?;
        let sigma_from = self.sigmas[step];
        let sigma_to = self.sigmas[step + 1];
        let sigma_up = (sigma_to * sigma_to * (sigma_from * sigma_from - sigma_to * sigma_to)
            / (sigma_from * sigma_from))
            .max(0.0)
            .sqrt();
        let sigma_down = (sigma_to * sigma_to - sigma_up * sigma_up).max(0.0).sqrt();
        let dt = sigma_down - sigma_from;
        let noise = noise.context("euler_ancestral requires per-step noise")?.to_vec_f32_lossy()?;
        if noise.len() != x.len() {
            anyhow::bail!("euler_ancestral noise length {} != sample {}", noise.len(), x.len());
        }
        let out: Vec<f32> = (0..x.len())
            .map(|i| x[i] + eps[i] * dt + noise[i] * sigma_up)
            .collect();
        Value::from_slice_f32(&out, &shape).map_err(Into::into)
    }

    fn scale_input(
        &self,
        step: usize,
        _num_steps: usize,
        sample: &Value,
    ) -> anyhow::Result<Option<Value>> {
        let factor = (self.sigmas[step] * self.sigmas[step] + 1.0).sqrt();
        let scaled: Vec<f32> = sample.to_vec_f32_lossy()?.iter().map(|&x| x / factor).collect();
        Ok(Some(Value::from_slice_f32(&scaled, sample.shape())?))
    }

    fn init_noise_sigma(&self) -> f32 {
        self.sigmas[0]
    }

    fn timesteps(&self) -> Option<Vec<f32>> {
        Some(self.timesteps.clone())
    }
}

/// DPM-Solver++ (2M) — a fast *multistep* deterministic scheduler and the default
/// sampler in most Stable Diffusion / ComfyUI workflows. Order-2 in log-SNR (λ)
/// space using the previous step's data prediction (`x0`), with a first-order step
/// at the start and a first-order final step (when `<15` steps or the final sigma
/// is zero, matching diffusers `final_sigmas_type="zero"`). Matches diffusers
/// `DPMSolverMultistepScheduler(algorithm_type="dpmsolver++", solver_type="midpoint")`.
/// Unlike Euler it does NOT scale the model input (`scale_model_input` is identity)
/// and its `init_noise_sigma` is 1.0 (the seed is unscaled).
#[derive(Debug)]
struct Dpmpp2m {
    /// Inference sigmas, descending, with a trailing `0.0`. Length `num_steps + 1`.
    sigmas: Vec<f32>,
    /// Per-step denoiser timesteps, length `num_steps`.
    timesteps: Vec<f32>,
    /// Previous step's data prediction (`x0`) for the multistep update. Reset at
    /// step 0 of each denoise loop; interior-mutable so `step` keeps `&self`.
    prev_x0: Mutex<Option<Vec<f32>>>,
}

impl Dpmpp2m {
    fn with_schedule(
        num_train_timesteps: usize,
        beta_start: f32,
        beta_end: f32,
        beta_schedule: &str,
        num_steps: usize,
        spacing: &str,
    ) -> anyhow::Result<Self> {
        if num_train_timesteps < 2 {
            anyhow::bail!("scheduler num_train_timesteps must be >= 2");
        }
        if num_steps == 0 || num_steps > num_train_timesteps {
            anyhow::bail!(
                "scheduler num_steps ({num_steps}) must be in 1..={num_train_timesteps}"
            );
        }
        if let Some(sigmas) =
            spacing_sigmas(spacing, num_train_timesteps, beta_start, beta_end, beta_schedule, num_steps)?
        {
            let train = training_sigmas(num_train_timesteps, beta_start, beta_end, beta_schedule)?;
            let timesteps = sigmas[..num_steps].iter().map(|&s| sigma_to_t(&train, s)).collect();
            return Ok(Self {
                sigmas,
                timesteps,
                prev_x0: Mutex::new(None),
            });
        }
        let denom = (num_train_timesteps - 1) as f32;
        let (lo, hi, square) = match beta_schedule {
            "linear" => (beta_start, beta_end, false),
            "scaled_linear" => (beta_start.sqrt(), beta_end.sqrt(), true),
            other => anyhow::bail!(
                "unsupported scheduler beta_schedule '{other}' (expected 'linear' or 'scaled_linear')"
            ),
        };
        let mut train = Vec::with_capacity(num_train_timesteps);
        let mut prod = 1.0f32;
        for i in 0..num_train_timesteps {
            let mut beta = lo + (hi - lo) * (i as f32) / denom;
            if square {
                beta *= beta;
            }
            prod *= 1.0 - beta;
            train.push(((1.0 - prod) / prod).sqrt());
        }
        // Timesteps: linspace(0, num_train-1, num_steps+1) rounded to int, reversed,
        // drop the last (the 0). Sigmas interpolate at those integer timesteps
        // (integer => exact lookup). Trailing 0 for final_sigmas_type="zero".
        let mut ts_int: Vec<usize> = (0..=num_steps)
            .map(|j| (j as f32 * denom / num_steps as f32).round_ties_even() as usize)
            .collect();
        ts_int.reverse();
        ts_int.pop();
        let timesteps: Vec<f32> = ts_int.iter().map(|&t| t as f32).collect();
        let mut sigmas: Vec<f32> = ts_int
            .iter()
            .map(|&t| train[t.min(num_train_timesteps - 1)])
            .collect();
        sigmas.push(0.0);
        Ok(Self {
            sigmas,
            timesteps,
            prev_x0: Mutex::new(None),
        })
    }
}

/// `alpha_t = 1/sqrt(sigma^2+1)`, `sigma_t = sigma * alpha_t` (diffusers convention).
fn dpm_alpha_sigma(sigma: f32) -> (f32, f32) {
    let alpha_t = 1.0 / (sigma * sigma + 1.0).sqrt();
    (alpha_t, sigma * alpha_t)
}

/// Training sigmas `((1-alpha_cumprod)/alpha_cumprod)^0.5` over the beta schedule.
fn training_sigmas(
    num_train_timesteps: usize,
    beta_start: f32,
    beta_end: f32,
    beta_schedule: &str,
) -> anyhow::Result<Vec<f32>> {
    let denom = (num_train_timesteps - 1) as f32;
    let (lo, hi, square) = match beta_schedule {
        "linear" => (beta_start, beta_end, false),
        "scaled_linear" => (beta_start.sqrt(), beta_end.sqrt(), true),
        other => anyhow::bail!(
            "unsupported scheduler beta_schedule '{other}' (expected 'linear' or 'scaled_linear')"
        ),
    };
    let mut out = Vec::with_capacity(num_train_timesteps);
    let mut prod = 1.0f32;
    for i in 0..num_train_timesteps {
        let mut beta = lo + (hi - lo) * (i as f32) / denom;
        if square {
            beta *= beta;
        }
        prod *= 1.0 - beta;
        out.push(((1.0 - prod) / prod).sqrt());
    }
    Ok(out)
}

/// Interpolate a diffusion timestep from a sigma, matching diffusers
/// `SchedulerMixin._sigma_to_t`. `train` holds the ascending training sigmas
/// (index = training timestep). Finds the bracketing training sigmas in log space
/// and linearly interpolates the (fractional) timestep. Used to recover the
/// denoiser timesteps for sigma-space schedules (Karras / exponential) where the
/// timesteps are not the sigma indices.
fn sigma_to_t(train: &[f32], sigma: f32) -> f32 {
    let log_sigma = sigma.max(1e-10).ln();
    let count = train.iter().filter(|&&s| s.max(1e-10).ln() <= log_sigma).count();
    let low_idx = count.saturating_sub(1).min(train.len().saturating_sub(2));
    let high_idx = low_idx + 1;
    let low = train[low_idx].max(1e-10).ln();
    let high = train[high_idx].max(1e-10).ln();
    let weight = ((low - log_sigma) / (low - high)).clamp(0.0, 1.0);
    (1.0 - weight) * low_idx as f32 + weight * high_idx as f32
}

/// Karras (rho=7) sigma schedule from the training sigma range, descending, with
/// a trailing `0.0`. Length `num_steps + 1`. Matches diffusers `_convert_to_karras`
/// (identical for Euler and DPM++ since both derive min/max from the full range).
fn karras_sigmas(
    num_train_timesteps: usize,
    beta_start: f32,
    beta_end: f32,
    beta_schedule: &str,
    num_steps: usize,
) -> anyhow::Result<Vec<f32>> {
    const RHO: f32 = 7.0;
    let train = training_sigmas(num_train_timesteps, beta_start, beta_end, beta_schedule)?;
    let sigma_min = train[0];
    let sigma_max = train[num_train_timesteps - 1];
    let min_inv = sigma_min.powf(1.0 / RHO);
    let max_inv = sigma_max.powf(1.0 / RHO);
    let mut sigmas = Vec::with_capacity(num_steps + 1);
    for k in 0..num_steps {
        let ramp = if num_steps > 1 {
            k as f32 / (num_steps - 1) as f32
        } else {
            0.0
        };
        sigmas.push((max_inv + ramp * (min_inv - max_inv)).powf(RHO));
    }
    sigmas.push(0.0);
    Ok(sigmas)
}

/// Exponential sigma schedule: `exp(linspace(log(sigma_max), log(sigma_min), n))`,
/// descending, trailing `0.0`. Same training-sigma min/max as Karras. Matches
/// diffusers `_convert_to_exponential`.
fn exponential_sigmas(
    num_train_timesteps: usize,
    beta_start: f32,
    beta_end: f32,
    beta_schedule: &str,
    num_steps: usize,
) -> anyhow::Result<Vec<f32>> {
    let train = training_sigmas(num_train_timesteps, beta_start, beta_end, beta_schedule)?;
    let log_min = train[0].ln();
    let log_max = train[num_train_timesteps - 1].ln();
    let mut sigmas = Vec::with_capacity(num_steps + 1);
    for k in 0..num_steps {
        let ramp = if num_steps > 1 {
            k as f32 / (num_steps - 1) as f32
        } else {
            0.0
        };
        sigmas.push((log_max + ramp * (log_min - log_max)).exp());
    }
    sigmas.push(0.0);
    Ok(sigmas)
}

/// Select the sigma schedule the spec requests: `"karras"`, `"exponential"`, or
/// the default `"linspace"`. Rejects the conflicting case where both Karras and
/// exponential are requested.
fn sigma_spacing(cfg: &SchedulerSpec) -> anyhow::Result<&'static str> {
    let karras = cfg.use_karras_sigmas.unwrap_or(false);
    let exponential = cfg.use_exponential_sigmas.unwrap_or(false);
    if karras && exponential {
        anyhow::bail!(
            "scheduler cannot set both use_karras_sigmas and use_exponential_sigmas"
        );
    }
    Ok(if karras {
        "karras"
    } else if exponential {
        "exponential"
    } else {
        "linspace"
    })
}

/// Precomputed sigmas for a non-linspace spacing (`karras`/`exponential`), or
/// `None` for the default `linspace` (which each scheduler builds itself).
fn spacing_sigmas(
    spacing: &str,
    num_train_timesteps: usize,
    beta_start: f32,
    beta_end: f32,
    beta_schedule: &str,
    num_steps: usize,
) -> anyhow::Result<Option<Vec<f32>>> {
    match spacing {
        "karras" => Ok(Some(karras_sigmas(
            num_train_timesteps, beta_start, beta_end, beta_schedule, num_steps,
        )?)),
        "exponential" => Ok(Some(exponential_sigmas(
            num_train_timesteps, beta_start, beta_end, beta_schedule, num_steps,
        )?)),
        "linspace" | "" => Ok(None),
        other => anyhow::bail!("unsupported sigma spacing '{other}' (karras/exponential/linspace)"),
    }
}

impl Scheduler for Dpmpp2m {
    fn step(
        &self,
        step: usize,
        num_steps: usize,
        sample: &Value,
        model_output: &Value,
    ) -> anyhow::Result<Value> {
        let shape = sample.shape().to_vec();
        let x = sample.to_vec_f32_lossy()?;
        let eps = model_output.to_vec_f32_lossy()?;
        if x.len() != eps.len() {
            anyhow::bail!("dpm++ sample/eps length mismatch: {} vs {}", x.len(), eps.len());
        }

        let sigma = self.sigmas[step];
        let (alpha_t0, sigma_t0) = dpm_alpha_sigma(sigma);
        // Data prediction (x0) from the epsilon output: (x - sigma_t*eps)/alpha_t.
        let x0: Vec<f32> = x
            .iter()
            .zip(&eps)
            .map(|(&xi, &ei)| (xi - sigma_t0 * ei) / alpha_t0)
            .collect();

        let s_next = self.sigmas[step + 1];
        let (a_t, sig_t) = dpm_alpha_sigma(s_next);
        let (a_s0, sig_s0) = dpm_alpha_sigma(sigma);
        let lam_t = a_t.ln() - sig_t.ln(); // +inf at the final step (sig_t == 0)
        let lam_s0 = a_s0.ln() - sig_s0.ln();
        let h = lam_t - lam_s0;
        let neg_expm1 = (-h).exp() - 1.0; // exp(-h) - 1  (== -1 at the final step)

        let mut prev = self
            .prev_x0
            .lock()
            .map_err(|_| anyhow::anyhow!("dpm++ scheduler state poisoned"))?;
        // Match diffusers `DPMSolverMultistepScheduler`: the final step drops to
        // the first-order update when `lower_order_final` applies. diffusers sets
        // that at the last step whenever `num_steps < 15` OR the final sigma is
        // zero (`final_sigmas_type="zero"`, the default this schedule uses). The
        // second-order update divides by the log-SNR step `h`, which is infinite
        // when the final sigma is zero — so skipping it there also avoids the
        // resulting non-finite latent.
        let lower_order_final = step + 1 == num_steps && (num_steps < 15 || s_next <= 0.0);
        // First step of the loop (prev cleared by reset) or the low-order final
        // step both use the first-order update.
        let first_order = lower_order_final || prev.is_none();

        let out: Vec<f32> = if first_order {
            x.iter()
                .zip(&x0)
                .map(|(&xi, &d0)| (sig_t / sig_s0) * xi - a_t * neg_expm1 * d0)
                .collect()
        } else {
            let prev_x0 = prev.as_ref().unwrap();
            let s_prev = self.sigmas[step - 1];
            let (a_s1, sig_s1) = dpm_alpha_sigma(s_prev);
            let lam_s1 = a_s1.ln() - sig_s1.ln();
            let h0 = lam_s0 - lam_s1;
            let r0 = h0 / h;
            x.iter()
                .enumerate()
                .map(|(i, &xi)| {
                    let d0 = x0[i];
                    let d1 = (1.0 / r0) * (x0[i] - prev_x0[i]);
                    (sig_t / sig_s0) * xi - a_t * neg_expm1 * d0 - 0.5 * a_t * neg_expm1 * d1
                })
                .collect()
        };
        *prev = Some(x0);
        drop(prev);
        Value::from_slice_f32(&out, &shape).map_err(Into::into)
    }

    fn reset(&self) {
        if let Ok(mut prev) = self.prev_x0.lock() {
            *prev = None;
        }
    }

    fn timesteps(&self) -> Option<Vec<f32>> {
        Some(self.timesteps.clone())
    }
}

impl PipelinePlan {
    fn from_spec(spec: &PipelineSpec, schedulers: &SchedulerRegistry) -> anyhow::Result<Self> {
        // A dual, hierarchically-nested AR pipeline (multi-decoder TTS) is
        // detected before the single-decoder AR path: its outer+inner decoders
        // are driven by a dedicated nested loop, not the flat AR decode driver.
        if let Some(stage) = nested_autoregressive_strategy(&spec.strategy) {
            return Self::nested_autoregressive(spec, stage);
        }
        // A composite whose stages contain an autoregressive decoder is treated
        // as an autoregressive text pipeline (unchanged legacy behavior). Pure
        // iterative / single-pass composites are a follow-up.
        if let Some(decoder) = autoregressive_decoder(&spec.strategy) {
            return Self::autoregressive(spec, decoder);
        }
        match spec.strategy.kind {
            PipelineStrategyKind::SinglePass => Self::single_pass(spec),
            PipelineStrategyKind::Iterative => Self::iterative(spec, schedulers),
            PipelineStrategyKind::Composite => Self::composite(spec),
            PipelineStrategyKind::Autoregressive => {
                anyhow::bail!("autoregressive strategy is missing its 'decoder' component")
            }
            PipelineStrategyKind::NestedAutoregressive => {
                anyhow::bail!(
                    "nested_autoregressive strategy is missing its 'outer'/'inner' decoders"
                )
            }
            PipelineStrategyKind::Other(ref value) => {
                anyhow::bail!("unsupported pipeline strategy kind '{value}'")
            }
        }
    }

    fn autoregressive(spec: &PipelineSpec, decoder: String) -> anyhow::Result<Self> {
        if !spec.models.contains_key(&decoder) {
            anyhow::bail!("pipeline decoder '{decoder}' is not declared in models");
        }
        let prompt_components = prompt_phase_components(spec, &decoder)?;
        let post_decode_components = post_decode_components(spec, &decoder)?;
        Ok(Self::Autoregressive(AutoregressivePlan {
            decoder,
            prompt_components,
            post_decode_components,
            dataflow: spec.dataflow.clone(),
        }))
    }

    /// Build a [`NestedAutoregressivePlan`] (multi-decoder TTS, DESIGN.md §20.3)
    /// from the `nested_autoregressive` strategy `nested` (which may be a
    /// top-level strategy or a composite stage). Validates the outer/inner
    /// decoders, the inner-loop depth, and the per-frame hidden binding.
    fn nested_autoregressive(
        spec: &PipelineSpec,
        nested: &PipelineStrategy,
    ) -> anyhow::Result<Self> {
        let outer = nested
            .outer
            .clone()
            .context("nested_autoregressive strategy is missing its 'outer' decoder")?;
        let inner = nested
            .inner
            .clone()
            .context("nested_autoregressive strategy is missing its 'inner' decoder")?;
        if !spec.models.contains_key(&outer) {
            anyhow::bail!(
                "nested_autoregressive outer decoder '{outer}' is not declared in models"
            );
        }
        if !spec.models.contains_key(&inner) {
            anyhow::bail!(
                "nested_autoregressive inner decoder '{inner}' is not declared in models"
            );
        }
        if outer == inner {
            anyhow::bail!(
                "nested_autoregressive 'outer' and 'inner' must be distinct decoders (both '{outer}')"
            );
        }
        let num_code_groups = nested
            .num_code_groups
            .context("nested_autoregressive strategy is missing 'num_code_groups'")?;
        if num_code_groups == 0 {
            anyhow::bail!("nested_autoregressive 'num_code_groups' must be greater than zero");
        }
        let max_frames = nested
            .max_tokens
            .context("nested_autoregressive strategy is missing 'max_tokens' (max audio frames)")?;
        if max_frames == 0 {
            anyhow::bail!(
                "nested_autoregressive 'max_tokens' (max frames) must be greater than zero"
            );
        }

        // The per-frame hidden binding is the dataflow edge feeding the inner
        // decoder's seed input from the outer decoder's hidden-state output.
        let inner_embeds_endpoint_edge = spec
            .dataflow
            .iter()
            .find(|edge| {
                endpoint_component(&edge.to) == Some(inner.as_str())
                    && endpoint_component(&edge.from) == Some(outer.as_str())
            })
            .with_context(|| {
                format!(
                    "nested_autoregressive needs a per-frame hidden binding: a dataflow edge \
                     '{outer}.last_hidden_state -> {inner}.inputs_embeds'"
                )
            })?;
        let (_, outer_hidden_output) = parse_endpoint(&inner_embeds_endpoint_edge.from)?;
        let (_, inner_embeds_input) = parse_endpoint(&inner_embeds_endpoint_edge.to)?;
        let outer_hidden_output = outer_hidden_output.to_string();
        let inner_embeds_input = inner_embeds_input.to_string();

        // The inner decoder threads its own per-code embedding on later steps;
        // its exact output port is resolved from the loaded session in the driver
        // (the sole non-logits, non-KV output), since sessions are not available
        // at plan-build time.

        // Optional pre-embedder driving the outer talker via `inputs_embeds`
        // (materialized codec-sum embedder) instead of `input_ids`. When set it
        // must be a declared model, distinct from the loop decoders, and wired to
        // the outer decoder by a dataflow edge
        // `{pre_embedder}.inputs_embeds -> {outer}.inputs_embeds`.
        let pre_embedder = match nested.pre_embedder.as_deref() {
            Some(name) => {
                if !spec.models.contains_key(name) {
                    anyhow::bail!(
                        "nested_autoregressive pre_embedder '{name}' is not declared in models"
                    );
                }
                if name == outer || name == inner {
                    anyhow::bail!(
                        "nested_autoregressive pre_embedder '{name}' must be distinct from the \
                         outer/inner decoders"
                    );
                }
                let edge = spec
                    .dataflow
                    .iter()
                    .find(|edge| {
                        endpoint_component(&edge.from) == Some(name)
                            && endpoint_component(&edge.to) == Some(outer.as_str())
                    })
                    .with_context(|| {
                        format!(
                            "nested_autoregressive pre_embedder '{name}' needs a per-step feed: a \
                             dataflow edge '{name}.inputs_embeds -> {outer}.inputs_embeds'"
                        )
                    })?;
                let (_, outer_input) = parse_endpoint(&edge.to)?;
                Some(PreEmbedderBinding {
                    component: name.to_string(),
                    outer_input: outer_input.to_string(),
                })
            }
            None => None,
        };
        let pre_embedder_component = pre_embedder.as_ref().map(|p| p.component.clone());

        // Prompt-phase (`prompt_only`) and post-decode (`final_only`) components,
        // treating both loop decoders as loop components (neither pre nor post).
        let mut prompt_components = Vec::new();
        let mut post_decode_components = Vec::new();
        for component in topological_components(spec)? {
            if component == outer || component == inner {
                continue;
            }
            // The pre-embedder is driven per-frame inside the outer loop, not as a
            // prompt/final stage — exclude it from phase classification.
            if pre_embedder_component.as_deref() == Some(component.as_str()) {
                continue;
            }
            match component_phase(spec, &component, &outer) {
                PhaseRunOn::PromptOnly => prompt_components.push(component),
                PhaseRunOn::FinalOnly => post_decode_components.push(component),
                PhaseRunOn::OnDemand => {}
                PhaseRunOn::EveryStep => anyhow::bail!(
                    "nested_autoregressive component '{component}' declares run_on: every_step, \
                     but only the outer/inner decoders may run inside the nested loop"
                ),
                PhaseRunOn::Other(value) => anyhow::bail!(
                    "unsupported phase '{value}' for pipeline component '{component}'"
                ),
            }
        }

        Ok(Self::NestedAutoregressive(NestedAutoregressivePlan {
            outer,
            inner,
            num_code_groups,
            max_frames,
            outer_hidden_output,
            inner_embeds_input,
            prompt_components,
            post_decode_components,
            pre_embedder,
            dataflow: spec.dataflow.clone(),
        }))
    }

    fn single_pass(spec: &PipelineSpec) -> anyhow::Result<Self> {
        let model = spec
            .strategy
            .model
            .clone()
            .context("single_pass strategy is missing its 'model' component")?;
        if !spec.models.contains_key(&model) {
            anyhow::bail!("pipeline model '{model}' is not declared in models");
        }
        // Single-pass has no loop and no final stage, so `every_step` and
        // `final_only` components would be silently dropped — reject them.
        let mut prompt_components = Vec::new();
        for component in topological_components(spec)? {
            if component == model {
                continue;
            }
            match component_phase(spec, &component, &model) {
                PhaseRunOn::PromptOnly => prompt_components.push(component),
                PhaseRunOn::OnDemand => {}
                PhaseRunOn::EveryStep | PhaseRunOn::FinalOnly => anyhow::bail!(
                    "component '{component}' declares a run_on phase unsupported by a single_pass \
                     pipeline (only prompt_only / on_demand components are allowed)"
                ),
                PhaseRunOn::Other(value) => anyhow::bail!(
                    "unsupported phase '{value}' for pipeline component '{component}'"
                ),
            }
        }
        Ok(Self::SinglePass(SinglePassPlan {
            model,
            prompt_components,
            dataflow: spec.dataflow.clone(),
        }))
    }

    /// Build a multi-stage composite plan (DESIGN.md §20). Reached only for a
    /// `kind: composite` strategy that has no autoregressive decoder stage (those
    /// route to [`Self::autoregressive`]); i.e. pure single-pass stage chains such
    /// as audio-to-audio codecs, encoder chains, and vocoder post-processing.
    fn composite(spec: &PipelineSpec) -> anyhow::Result<Self> {
        if spec.strategy.stages.is_empty() {
            anyhow::bail!("composite pipeline strategy declares no stages");
        }
        let mut stages = Vec::with_capacity(spec.strategy.stages.len());
        let mut seen_names = BTreeSet::new();
        for stage in &spec.strategy.stages {
            if !seen_names.insert(stage.name.clone()) {
                anyhow::bail!("composite stage name '{}' is not unique", stage.name);
            }
            let kind = match stage.strategy.kind {
                PipelineStrategyKind::SinglePass => {
                    let model = stage.strategy.model.clone().with_context(|| {
                        format!("composite stage '{}' (single_pass) is missing 'model'", stage.name)
                    })?;
                    if !spec.models.contains_key(&model) {
                        anyhow::bail!(
                            "composite stage '{}' model '{model}' is not declared in models",
                            stage.name
                        );
                    }
                    CompositeStageKind::SinglePass { model }
                }
                PipelineStrategyKind::Iterative => anyhow::bail!(
                    "composite iterative stage '{}' is not yet supported (single-pass stages only)",
                    stage.name
                ),
                PipelineStrategyKind::Autoregressive
                | PipelineStrategyKind::Composite
                | PipelineStrategyKind::NestedAutoregressive => {
                    anyhow::bail!(
                        "composite stage '{}' has an unsupported nested strategy kind for a \
                         non-autoregressive composite",
                        stage.name
                    )
                }
                PipelineStrategyKind::Other(ref value) => anyhow::bail!(
                    "composite stage '{}' has unsupported strategy kind '{value}'",
                    stage.name
                ),
            };
            stages.push(CompositeStage {
                name: stage.name.clone(),
                kind,
            });
        }
        Ok(Self::Composite(CompositePlan {
            stages,
            dataflow: spec.dataflow.clone(),
        }))
    }

    fn iterative(spec: &PipelineSpec, schedulers: &SchedulerRegistry) -> anyhow::Result<Self> {
        let denoiser = spec
            .strategy
            .denoiser
            .clone()
            .context("iterative strategy is missing its 'denoiser' component")?;
        if !spec.models.contains_key(&denoiser) {
            anyhow::bail!("pipeline denoiser '{denoiser}' is not declared in models");
        }
        let num_steps = spec
            .strategy
            .num_steps
            .context("iterative strategy is missing 'num_steps'")?;
        if num_steps == 0 {
            anyhow::bail!("iterative strategy 'num_steps' must be greater than zero");
        }
        let start_step = spec.strategy.start_step.unwrap_or(0);
        if start_step >= num_steps {
            anyhow::bail!(
                "iterative strategy 'start_step' ({start_step}) must be less than 'num_steps' ({num_steps})"
            );
        }

        // Classifier-free guidance normally requires a declared conditioning
        // input to zero on the unconditional pass. Discrete language diffusion
        // (`masked_diffusion`) is the exception: its unconditional pass re-masks
        // the prompt of the loop-carried sample (via `Scheduler::cfg_uncond_sample`),
        // so it needs no conditioning port.
        let guidance_active = spec.strategy.guidance_scale.is_some_and(|s| s != 1.0);
        let scheduler_supplies_uncond = spec
            .strategy
            .scheduler_config
            .as_ref()
            .is_some_and(|scheduler| scheduler.kind == "masked_diffusion");
        if guidance_active
            && spec.strategy.cfg_conditioning_input.is_none()
            && !scheduler_supplies_uncond
        {
            anyhow::bail!(
                "classifier-free guidance (guidance_scale != 1.0) requires \
                 'cfg_conditioning_input' naming the denoiser conditioning port to zero on the \
                 unconditional pass"
            );
        }

        // Loop-carried edges are the denoiser's self-referential dataflow edges.
        let mut loop_edges = Vec::new();
        for edge in &spec.dataflow {
            let (from_component, from_port) = parse_endpoint(&edge.from)?;
            let (to_component, to_port) = parse_endpoint(&edge.to)?;
            if from_component == denoiser && to_component == denoiser {
                loop_edges.push((from_port.to_string(), to_port.to_string()));
            }
        }

        // The CFG conditioning port and the loop-carried sample port must be
        // distinct: on the unconditional pass the conditioning is replaced, and
        // a scheduler (e.g. Euler) may also override the loop input, so a shared
        // port would make the two overrides clobber each other.
        if let Some(cfg_port) = &spec.strategy.cfg_conditioning_input
            && guidance_active
            && loop_edges.iter().any(|(_, in_port)| in_port == cfg_port)
        {
            anyhow::bail!(
                "cfg_conditioning_input '{cfg_port}' must not also be a loop-carried input \
                 port: the unconditional conditioning override would clobber the loop sample"
            );
        }

        // Non-decoder components: prompt-phase (run once before the loop) and
        // final-phase (run once after the loop).
        let mut prompt_components = Vec::new();
        let mut final_components = Vec::new();
        for component in topological_components(spec)? {
            if component == denoiser {
                continue;
            }
            match component_phase(spec, &component, &denoiser) {
                PhaseRunOn::PromptOnly => prompt_components.push(component),
                PhaseRunOn::FinalOnly => final_components.push(component),
                PhaseRunOn::OnDemand => {}
                PhaseRunOn::EveryStep => anyhow::bail!(
                    "component '{component}' declares run_on: every_step, but running a \
                     non-denoiser component inside the iterative loop is not yet supported"
                ),
                PhaseRunOn::Other(value) => anyhow::bail!(
                    "unsupported phase '{value}' for pipeline component '{component}'"
                ),
            }
        }

        Ok(Self::Iterative(Box::new(IterativePlan {
            denoiser,
            num_steps,
            guidance_scale: spec.strategy.guidance_scale,
            prompt_components,
            final_components,
            loop_edges,
            timestep_input: spec.strategy.timestep_input.clone(),
            start_step,
            timesteps: spec.strategy.timesteps.clone(),
            scheduler: build_scheduler(spec.strategy.scheduler_config.as_ref(), num_steps, schedulers)?,
            cfg_conditioning_input: spec.strategy.cfg_conditioning_input.clone(),
            dataflow: spec.dataflow.clone(),
            scheduler_spec: spec.strategy.scheduler_config.clone(),
            scheduler_registry: schedulers.clone(),
        })))
    }

    fn autoregressive_plan(&self) -> anyhow::Result<&AutoregressivePlan> {
        match self {
            Self::Autoregressive(plan) => Ok(plan),
            _ => anyhow::bail!("pipeline strategy is not autoregressive"),
        }
    }

    fn dataflow(&self) -> &[DataflowEdge] {
        match self {
            Self::Autoregressive(plan) => &plan.dataflow,
            Self::NestedAutoregressive(plan) => &plan.dataflow,
            Self::SinglePass(plan) => &plan.dataflow,
            Self::Iterative(plan) => &plan.dataflow,
            Self::Composite(plan) => &plan.dataflow,
        }
    }

    fn edges_to_component<'a>(
        &'a self,
        component: &'a str,
    ) -> impl Iterator<Item = &'a DataflowEdge> + 'a {
        self.dataflow()
            .iter()
            .filter(move |edge| endpoint_component(&edge.to) == Some(component))
    }
}

/// Collect the `prompt_only`-phase components (everything except `primary`
/// defaults to prompt-phase), rejecting unsupported phase strings.
fn prompt_phase_components(spec: &PipelineSpec, primary: &str) -> anyhow::Result<Vec<String>> {
    let mut prompt_components = Vec::new();
    for component in topological_components(spec)? {
        if component == primary {
            continue;
        }
        match component_phase(spec, &component, primary) {
            PhaseRunOn::PromptOnly => prompt_components.push(component),
            PhaseRunOn::EveryStep | PhaseRunOn::OnDemand | PhaseRunOn::FinalOnly => {}
            PhaseRunOn::Other(value) => {
                anyhow::bail!("unsupported phase '{value}' for pipeline component '{component}'")
            }
        }
    }
    Ok(prompt_components)
}

/// Collect the `final_only`-phase components in dataflow order: single-pass
/// stages that run **once after** the AR decode loop completes (the TTS vocoder
/// shape from DESIGN.md §20). Kept separate from [`prompt_phase_components`] so
/// the decode loop's generated code tokens (exposed as `{decoder}.output_ids`)
/// can be routed into them before they run.
fn post_decode_components(spec: &PipelineSpec, decoder: &str) -> anyhow::Result<Vec<String>> {
    let mut post = Vec::new();
    for component in topological_components(spec)? {
        if component == decoder {
            continue;
        }
        if let PhaseRunOn::FinalOnly = component_phase(spec, &component, decoder) {
            post.push(component);
        }
    }
    Ok(post)
}

/// Build a DDIM scheduler from the declared config, or `None` when no scheduler
/// is configured. Delegates to the registry so custom scheduler kinds work.
fn build_scheduler(
    config: Option<&SchedulerSpec>,
    num_steps: usize,
    registry: &SchedulerRegistry,
) -> anyhow::Result<Option<Arc<dyn Scheduler>>> {
    let Some(cfg) = config else {
        return Ok(None);
    };
    Ok(Some(registry.build(cfg, num_steps)?))
}

fn autoregressive_decoder(strategy: &PipelineStrategy) -> Option<String> {
    match strategy.kind {
        PipelineStrategyKind::Autoregressive => strategy.decoder.clone(),
        PipelineStrategyKind::Composite => strategy
            .stages
            .iter()
            .find_map(|stage| autoregressive_decoder(&stage.strategy)),
        PipelineStrategyKind::Iterative
        | PipelineStrategyKind::SinglePass
        | PipelineStrategyKind::NestedAutoregressive
        | PipelineStrategyKind::Other(_) => None,
    }
}

/// Find the `nested_autoregressive` strategy in a pipeline: either the top-level
/// strategy or a stage of a composite. Returns the strategy carrying the
/// `outer` / `inner` / `num_code_groups` fields.
fn nested_autoregressive_strategy(strategy: &PipelineStrategy) -> Option<&PipelineStrategy> {
    match strategy.kind {
        PipelineStrategyKind::NestedAutoregressive => Some(strategy),
        PipelineStrategyKind::Composite => strategy
            .stages
            .iter()
            .find_map(|stage| nested_autoregressive_strategy(&stage.strategy)),
        _ => None,
    }
}

fn component_phase(spec: &PipelineSpec, component: &str, decoder: &str) -> PhaseRunOn {
    spec.phases
        .get(component)
        .map(|phase| phase.run_on.clone())
        .unwrap_or_else(|| {
            if component == decoder {
                PhaseRunOn::EveryStep
            } else {
                PhaseRunOn::PromptOnly
            }
        })
}

fn topological_components(spec: &PipelineSpec) -> anyhow::Result<Vec<String>> {
    let mut remaining = spec.models.keys().cloned().collect::<BTreeSet<_>>();
    let mut ordered = Vec::new();
    while !remaining.is_empty() {
        let ready = remaining
            .iter()
            .find(|component| {
                spec.dataflow.iter().all(|edge| {
                    let to = endpoint_component(&edge.to);
                    let from = endpoint_component(&edge.from);
                    // The edge does not gate `component` when: it does not feed
                    // `component`; it is a self-edge (loop-carried, resolved
                    // temporally, not an ordering dependency); or its source is
                    // already ordered.
                    to != Some(component.as_str())
                        || from == Some(component.as_str())
                        || from.is_some_and(|f| !remaining.contains(f))
                })
            })
            .cloned();
        let Some(component) = ready else {
            anyhow::bail!("pipeline dataflow contains a cycle");
        };
        remaining.remove(&component);
        ordered.push(component);
    }
    Ok(ordered)
}

fn parse_endpoint(endpoint: &str) -> anyhow::Result<(&str, &str)> {
    endpoint
        .split_once('.')
        .filter(|(component, port)| !component.is_empty() && !port.is_empty())
        .with_context(|| format!("pipeline endpoint must be component.port: {endpoint}"))
}

fn endpoint_component(endpoint: &str) -> Option<&str> {
    parse_endpoint(endpoint)
        .ok()
        .map(|(component, _)| component)
}

/// Locate a named session output by index and return a reference to its value.
///
/// With `contains == false` the name must match exactly; with `contains == true`
/// an exact match is preferred but a case-insensitive substring match (e.g. a
/// prefixed `logits`) is accepted as a fallback, mirroring the decode helpers.
fn named_output<'a>(
    session: &Session,
    outputs: &'a [Value],
    name: &str,
    contains: bool,
) -> anyhow::Result<&'a Value> {
    let index = session
        .output_names()
        .iter()
        .position(|out| out == name)
        .or_else(|| {
            if contains {
                let needle = name.to_ascii_lowercase();
                session
                    .output_names()
                    .iter()
                    .position(|out| out.to_ascii_lowercase().contains(&needle))
            } else {
                None
            }
        })
        .with_context(|| format!("model did not expose output '{name}'"))?;
    outputs
        .get(index)
        .with_context(|| format!("output '{name}' index was out of range"))
}

/// Argmax over the last sequence row of a logits tensor (`[V]`, `[S, V]`, or
/// `[1, S, V]`), returning the winning vocabulary index. Ties take the lowest
/// index, matching greedy decoding.
fn argmax_last_row(logits: &Value) -> anyhow::Result<i64> {
    let shape = logits.shape();
    let data = logits
        .to_vec_f32_lossy()
        .map_err(|e| anyhow::anyhow!("failed to read logits tensor: {e}"))?;
    let vocab = match shape {
        [vocab] if *vocab > 0 => *vocab as usize,
        [seq, vocab] if *seq > 0 && *vocab > 0 => *vocab as usize,
        [batch, seq, vocab] if *batch == 1 && *seq > 0 && *vocab > 0 => *vocab as usize,
        other => anyhow::bail!("unsupported logits tensor shape: {other:?}"),
    };
    let start = data.len() - vocab;
    let row = &data[start..];
    let mut best = 0usize;
    for (i, &value) in row.iter().enumerate() {
        if value > row[best] {
            best = i;
        }
    }
    Ok(best as i64)
}

/// Slice the last sequence position of a hidden-state tensor (`[H]`, `[S, H]`,
/// or `[1, S, H]`) into a `[1, 1, H]` `float32` seed for the inner decoder.
fn last_position_hidden(hidden: &Value) -> anyhow::Result<Value> {
    let shape = hidden.shape();
    let data = hidden
        .to_vec_f32_lossy()
        .map_err(|e| anyhow::anyhow!("failed to read hidden-state tensor: {e}"))?;
    let hidden_dim = match shape {
        [h] if *h > 0 => *h as usize,
        [seq, h] if *seq > 0 && *h > 0 => *h as usize,
        [batch, seq, h] if *batch == 1 && *seq > 0 && *h > 0 => *h as usize,
        other => anyhow::bail!("unsupported hidden-state tensor shape: {other:?}"),
    };
    let start = data.len() - hidden_dim;
    Value::from_slice_f32(&data[start..], &[1, 1, hidden_dim as i64])
        .map_err(|e| anyhow::anyhow!("failed to build inner seed embedding: {e}"))
}

/// A [`PreEmbedderBinding`] resolved against its loaded session for driving —
/// the codec-sum pre-embedder that materializes the outer talker's per-step
/// `inputs_embeds` from the previous frame's codes.
struct ResolvedPreEmbedder<'a> {
    /// Loaded pre-embedder session.
    session: &'a Session,
    /// Outer decoder input port fed the per-step embeddings (`inputs_embeds`).
    outer_input: String,
    /// Pre-embedder input receiving the previous frame's codes (int64 `[1, G]`).
    frame_codes_input: String,
    /// Optional trailing-text input; fed zeros for now (documented follow-up).
    text_embed_input: Option<String>,
    /// Embedding hidden size for the emitted `inputs_embeds` / zero `text_embed`.
    hidden: usize,
}

/// Build the outer talker's per-step `inputs_embeds` by running the codec-sum
/// pre-embedder over one frame's `frame_codes` (`[outer_code_0, inner_code_1,
/// ..., inner_code_{G-1}]`). Returns a `[1, 1, hidden]` embedding.
///
/// TODO(any-to-any): the pre-embedder's `text_embed` (trailing-text
/// conditioning) input is fed a zero `[1, 1, hidden]` tensor for now. Real
/// trailing-text threading (and full prefill-embeds materialization on frame 0)
/// is a deliberate follow-up.
fn run_pre_embedder(pre: &ResolvedPreEmbedder<'_>, frame_codes: &[i64]) -> anyhow::Result<Value> {
    let mut inputs: Vec<(String, Value)> = Vec::with_capacity(2);
    inputs.push((
        pre.frame_codes_input.clone(),
        Value::from_slice_i64(frame_codes, &[1, frame_codes.len() as i64])?,
    ));
    if let Some(name) = &pre.text_embed_input {
        let dtype = pre
            .session
            .inputs()
            .iter()
            .find(|info| &info.name == name)
            .map(|info| info.dtype)
            .unwrap_or(DataType::Float32);
        let zeros = vec![0.0f32; pre.hidden];
        inputs.push((
            name.clone(),
            Value::from_f32_slice_as(&zeros, &[1, 1, pre.hidden as i64], dtype)
                .map_err(|e| anyhow::anyhow!("failed to build zero text_embed: {e}"))?,
        ));
    }
    let refs = inputs
        .iter()
        .map(|(name, value)| (name.as_str(), value))
        .collect::<Vec<_>>();
    let outputs = pre
        .session
        .run(&refs)
        .map_err(|e| anyhow::anyhow!("ORT pre-embedder run failed: {e}"))?;
    let index = pre
        .session
        .output_names()
        .iter()
        .position(|name| name == "inputs_embeds")
        .or_else(|| {
            pre.session
                .output_names()
                .iter()
                .position(|name| name.to_ascii_lowercase().ends_with("inputs_embeds"))
        })
        .unwrap_or(0);
    let value = outputs
        .get(index)
        .context("pre-embedder produced no inputs_embeds output")?;
    clone_value(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_genai_metadata::{PhaseConfig, PipelineComponentSpec, PipelineStrategyStage};
    use std::collections::BTreeMap;

    #[test]
    fn ddim_step_matches_hand_computed_closed_form() {
        // num_train=2, beta_start=beta_end=0.5 => betas=[0.5,0.5],
        // alphas=[0.5,0.5], alpha_cumprod=[0.5,0.25].
        // num_steps=1 => timestep t=0 => a_t=0.5, a_prev=1.0 (final step).
        //   x0_hat = (x - sqrt(0.5)*e) / sqrt(0.5)
        //   next   = sqrt(1)*x0_hat + sqrt(0)*e = x0_hat
        let sched = DdimSchedule::with_schedule(2, 0.5, 0.5, "linear", 1).expect("schedule builds");
        // x=1, e=0 -> next = 1/sqrt(0.5) = sqrt(2) ~= 1.41421356
        let n0 = sched.step(0, &[1.0], &[0.0]).unwrap();
        assert!((n0[0] - std::f32::consts::SQRT_2).abs() < 1e-5, "{}", n0[0]);
        // x=1, e=1 -> x0_hat = (1 - sqrt(0.5))/sqrt(0.5) = sqrt(2) - 1 ~= 0.41421356
        let n1 = sched.step(0, &[1.0], &[1.0]).unwrap();
        assert!((n1[0] - (std::f32::consts::SQRT_2 - 1.0)).abs() < 1e-5, "{}", n1[0]);
    }

    #[test]
    fn ddim_new_rejects_invalid_step_counts() {
        assert!(DdimSchedule::with_schedule(1, 0.1, 0.2, "linear", 1).is_err()); // num_train < 2
        assert!(DdimSchedule::with_schedule(4, 0.1, 0.2, "linear", 0).is_err()); // num_steps == 0
        assert!(DdimSchedule::with_schedule(4, 0.1, 0.2, "linear", 5).is_err()); // num_steps > num_train
    }

    #[test]
    fn dpmpp_timesteps_match_diffusers_linspace() {
        // Classic Stable Diffusion 1.x schedule (1000 train steps, scaled_linear).
        // diffusers `DPMSolverMultistepScheduler(timestep_spacing="linspace")`
        // uses `linspace(0, num_train-1, num_steps+1).round()[::-1][:-1]`.
        let num_train = 1000usize;
        let num_steps = 25usize;
        let sched = Dpmpp2m::with_schedule(num_train, 0.00085, 0.012, "scaled_linear", num_steps, "")
            .expect("schedule builds");
        let timesteps = sched.timesteps().expect("dpm++ exposes timesteps");
        let denom = (num_train - 1) as f32;
        let mut expected: Vec<f32> = (0..=num_steps)
            .map(|j| (j as f32 * denom / num_steps as f32).round_ties_even())
            .collect();
        expected.reverse();
        expected.pop();
        assert_eq!(timesteps.len(), num_steps);
        assert!((timesteps[0] - 999.0).abs() < 1e-3, "first timestep {}", timesteps[0]);
        for (got, want) in timesteps.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-3, "timestep {got} != {want}");
        }
    }

    #[test]
    fn ddim_exposes_descending_integer_timesteps() {
        let sched =
            DdimSchedule::with_schedule(1000, 0.00085, 0.012, "scaled_linear", 4).expect("builds");
        // step_ratio = 250, ascending = [0, 250, 500, 750], reversed for inference.
        assert_eq!(sched.timesteps(), Some(vec![750.0, 500.0, 250.0, 0.0]));
    }

    #[test]
    fn masked_diffusion_random_unmasks_all_and_never_emits_mask() {
        // MDLM-style ancestral unmasking: by the final step every masked position
        // must be filled, the mask token must never be emitted (even though it has
        // the largest raw logit here), the prompt prefix is preserved, and the run
        // is deterministic.
        let vocab = 5usize;
        let mask_id = 4i64;
        let seq = 6usize;
        let prompt_len = 2usize;
        let num_steps = 4usize;

        // Sharp logits: the mask token has the highest logit (must be excluded),
        // and each position's highest *non-mask* logit is a distinct token.
        let mut logits = vec![0f32; seq * vocab];
        for pos in 0..seq {
            logits[pos * vocab + mask_id as usize] = 100.0;
            logits[pos * vocab + (pos % 4)] = 10.0;
        }
        let logits_value =
            Value::from_slice_f32(&logits, &[1, seq as i64, vocab as i64]).expect("logits");

        let sched = MaskedDiffusion {
            mask_token_id: mask_id,
            temperature: 0.0, // greedy token choice => deterministic, random unmask order
            block_length: None,
            remasking: Remasking::Random,
            generation_start: Mutex::new(None),
        };

        let seed = vec![1i64, 2, mask_id, mask_id, mask_id, mask_id];
        let run = |sched: &MaskedDiffusion| -> Vec<i64> {
            sched.reset();
            let mut value = Value::from_slice_i64(&seed, &[1, seq as i64]).expect("seed");
            for step in 0..num_steps {
                value = sched.step(step, num_steps, &value, &logits_value).expect("step");
            }
            value.to_vec_i64().expect("tokens")
        };

        let out = run(&sched);
        assert_eq!(&out[..prompt_len], &[1, 2], "prompt prefix preserved");
        for (pos, &tok) in out.iter().enumerate() {
            assert_ne!(tok, mask_id, "position {pos} still masked / emitted the mask token");
        }
        for pos in prompt_len..seq {
            assert_eq!(out[pos], (pos % 4) as i64, "position {pos} token");
        }
        assert_eq!(run(&sched), out, "ancestral sampling is deterministic");
    }

    #[test]
    fn masked_diffusion_rejects_unknown_remasking() {
        let registry = SchedulerRegistry::default();
        let spec = SchedulerSpec {
            kind: "masked_diffusion".to_string(),
            mask_token_id: Some(4),
            remasking: Some("nonsense".to_string()),
            ..SchedulerSpec::default()
        };
        assert!(registry.build(&spec, 4).is_err());
    }

    #[test]
    fn dpmpp_final_step_stays_finite_with_zero_final_sigma() {
        // With >= 15 steps and a zero final sigma (final_sigmas_type="zero"), the
        // last step must drop to the first-order update; the second-order update
        // divides by an infinite log-SNR step at sigma=0 and would emit NaN/inf.
        let num_steps = 20usize;
        let sched = Dpmpp2m::with_schedule(1000, 0.00085, 0.012, "scaled_linear", num_steps, "")
            .expect("schedule builds");
        sched.reset();
        let mut sample = Value::from_slice_f32(&[1.0, -0.5, 0.25], &[3]).unwrap();
        for step in 0..num_steps {
            let eps = Value::from_slice_f32(&[0.3, -0.2, 0.1], &[3]).unwrap();
            sample = sched.step(step, num_steps, &sample, &eps).unwrap();
        }
        assert!(
            sample.to_vec_f32().unwrap().iter().all(|value| value.is_finite()),
            "final dpm++ sample must be finite"
        );
    }

    fn component(role: &str) -> PipelineComponentSpec {
        PipelineComponentSpec {
            filename: format!("{role}.onnx"),
            role: role.to_string(),
            device_preference: None,
            tokenizer: None,
        }
    }

    #[test]
    fn explicit_native_backend_is_rejected_before_loading_pipeline_models() {
        let error = PipelineEngine::from_dir_with_config(
            Path::new("does-not-need-to-exist"),
            EngineConfig {
                decode_backend: EngineDecodeBackend::Native,
                ..EngineConfig::default()
            },
        )
        .err()
        .expect("native pipeline backend must be rejected");
        assert!(
            error
                .to_string()
                .contains("native backend not supported for pipeline models")
        );
    }

    #[cfg(feature = "native-backend")]
    #[test]
    fn auto_backend_rejects_pipeline_component_requiring_native() -> anyhow::Result<()> {
        use onnx_runtime_loader::proto::{
            ModelProto,
            onnx::{GraphProto, NodeProto, OperatorSetIdProto},
        };
        use prost::Message;

        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/test-fixtures/pipeline-native-backend-rejection");
        std::fs::create_dir_all(&root)?;
        let model = ModelProto {
            opset_import: vec![OperatorSetIdProto {
                domain: "pkg.nxrt".to_string(),
                version: 1,
            }],
            graph: Some(GraphProto {
                node: vec![NodeProto {
                    domain: "pkg.nxrt".to_string(),
                    op_type: "BlockQuantizedMatMul".to_string(),
                    ..NodeProto::default()
                }],
                ..GraphProto::default()
            }),
            ..ModelProto::default()
        };
        std::fs::write(root.join("decoder.onnx"), model.encode_to_vec())?;
        std::fs::write(
            root.join("inference_metadata.yaml"),
            r#"
pipeline:
  models:
    decoder:
      filename: decoder.onnx
      type: decoder
  dataflow: []
  strategy:
    kind: autoregressive
    decoder: decoder
"#,
        )?;

        let error = PipelineEngine::from_dir_with_config(&root, EngineConfig::default())
            .err()
            .expect("Auto must reject native-only pipeline components");
        assert!(
            error
                .to_string()
                .contains("native backend not supported for pipeline models")
        );
        Ok(())
    }

    #[test]
    fn plan_routes_prompt_encoder_outputs_to_decoder_inputs() -> anyhow::Result<()> {
        let spec = PipelineSpec {
            models: BTreeMap::from([
                ("vision_encoder".to_string(), component("encoder")),
                ("decoder".to_string(), component("decoder")),
            ]),
            dataflow: vec![DataflowEdge {
                from: "vision_encoder.image_features".to_string(),
                to: "decoder.encoder_hidden_states".to_string(),
                dtype: Some("fp32".to_string()),
                device_transfer: Some(false),
            }],
            strategy: PipelineStrategy {
                kind: PipelineStrategyKind::Composite,
                decoder: None,
                max_tokens: None,
                stop_conditions: None,
                kv_cache: None,
                speculative: None,
                model: None,
                batching: None,
                denoiser: None,
                scheduler: None,
                num_steps: None,
                timestep_input: None,
                timesteps: None,
                start_step: None,
                scheduler_config: None,
                cfg_conditioning_input: None,
                guidance_scale: None,
                state: None,
                outer: None,
                inner: None,
                num_code_groups: None,
                pre_embedder: None,
                stages: vec![
                    PipelineStrategyStage {
                        name: "encode".to_string(),
                        strategy: Box::new(PipelineStrategy {
                            kind: PipelineStrategyKind::SinglePass,
                            decoder: None,
                            max_tokens: None,
                            stop_conditions: None,
                            kv_cache: None,
                            speculative: None,
                            model: Some("vision_encoder".to_string()),
                            batching: None,
                            denoiser: None,
                            scheduler: None,
                            num_steps: None,
                            timestep_input: None,
                            timesteps: None,
                            start_step: None,
                            scheduler_config: None,
                            cfg_conditioning_input: None,
                            guidance_scale: None,
                            state: None,
                            outer: None,
                            inner: None,
                            num_code_groups: None,
                            pre_embedder: None,
                            stages: vec![],
                        }),
                        run_on: Some(PhaseRunOn::PromptOnly),
                    },
                    PipelineStrategyStage {
                        name: "decode".to_string(),
                        strategy: Box::new(PipelineStrategy {
                            kind: PipelineStrategyKind::Autoregressive,
                            decoder: Some("decoder".to_string()),
                            max_tokens: None,
                            stop_conditions: None,
                            kv_cache: None,
                            speculative: None,
                            model: None,
                            batching: None,
                            denoiser: None,
                            scheduler: None,
                            num_steps: None,
                            timestep_input: None,
                            timesteps: None,
                            start_step: None,
                            scheduler_config: None,
                            cfg_conditioning_input: None,
                            guidance_scale: None,
                            state: None,
                            outer: None,
                            inner: None,
                            num_code_groups: None,
                            pre_embedder: None,
                            stages: vec![],
                        }),
                        run_on: Some(PhaseRunOn::EveryStep),
                    },
                ],
            },
            phases: BTreeMap::from([
                (
                    "vision_encoder".to_string(),
                    PhaseConfig {
                        run_on: PhaseRunOn::PromptOnly,
                    },
                ),
                (
                    "decoder".to_string(),
                    PhaseConfig {
                        run_on: PhaseRunOn::EveryStep,
                    },
                ),
            ]),
            vision: None,
        };

        let plan = PipelinePlan::from_spec(&spec, &SchedulerRegistry::builtin())?;
        let ar = plan.autoregressive_plan()?;
        assert_eq!(ar.prompt_components, ["vision_encoder"]);
        assert_eq!(ar.decoder, "decoder");
        let routed = plan.edges_to_component("decoder").collect::<Vec<_>>();
        assert_eq!(routed.len(), 1);
        assert_eq!(
            parse_endpoint(&routed[0].to)?,
            ("decoder", "encoder_hidden_states")
        );
        assert_eq!(routed[0].from, "vision_encoder.image_features");
        Ok(())
    }

    fn bare_strategy(kind: PipelineStrategyKind) -> PipelineStrategy {
        PipelineStrategy {
            kind,
            decoder: None,
            max_tokens: None,
            stop_conditions: None,
            kv_cache: None,
            speculative: None,
            model: None,
            batching: None,
            denoiser: None,
            scheduler: None,
            num_steps: None,
            timestep_input: None,
            timesteps: None,
            start_step: None,
            scheduler_config: None,
            cfg_conditioning_input: None,
            guidance_scale: None,
            state: None,
            outer: None,
            inner: None,
            num_code_groups: None,
            pre_embedder: None,
            stages: vec![],
        }
    }

    fn single_pass_stage(name: &str, model: &str) -> PipelineStrategyStage {
        PipelineStrategyStage {
            name: name.to_string(),
            strategy: Box::new(PipelineStrategy {
                model: Some(model.to_string()),
                ..bare_strategy(PipelineStrategyKind::SinglePass)
            }),
            run_on: None,
        }
    }

    #[test]
    fn plan_builds_composite_single_pass_stages() -> anyhow::Result<()> {
        // Audio-to-audio codec: encoder -> decoder, both single-pass stages.
        let spec = PipelineSpec {
            models: BTreeMap::from([
                ("encoder".to_string(), component("encoder")),
                ("decoder".to_string(), component("decoder")),
            ]),
            dataflow: vec![DataflowEdge {
                from: "encoder.codes".to_string(),
                to: "decoder.codes".to_string(),
                dtype: Some("int64".to_string()),
                device_transfer: Some(false),
            }],
            strategy: PipelineStrategy {
                stages: vec![
                    single_pass_stage("encode", "encoder"),
                    single_pass_stage("decode", "decoder"),
                ],
                ..bare_strategy(PipelineStrategyKind::Composite)
            },
            phases: BTreeMap::new(),
            vision: None,
        };

        let plan = PipelinePlan::from_spec(&spec, &SchedulerRegistry::builtin())?;
        match &plan {
            PipelinePlan::Composite(composite) => {
                assert_eq!(composite.stages.len(), 2);
                assert_eq!(composite.stages[0].name, "encode");
                assert_eq!(composite.stages[1].name, "decode");
                assert!(matches!(
                    &composite.stages[0].kind,
                    CompositeStageKind::SinglePass { model } if model == "encoder"
                ));
                assert!(matches!(
                    &composite.stages[1].kind,
                    CompositeStageKind::SinglePass { model } if model == "decoder"
                ));
            }
            other => panic!("expected a Composite plan, got {other:?}"),
        }
        // Dataflow is preserved so the decoder stage reads the encoder's output.
        let routed = plan.edges_to_component("decoder").collect::<Vec<_>>();
        assert_eq!(routed.len(), 1);
        assert_eq!(routed[0].from, "encoder.codes");
        Ok(())
    }

    #[test]
    fn composite_iterative_stage_is_rejected_for_now() {
        let spec = PipelineSpec {
            models: BTreeMap::from([("encoder".to_string(), component("encoder"))]),
            dataflow: vec![],
            strategy: PipelineStrategy {
                stages: vec![PipelineStrategyStage {
                    name: "loop".to_string(),
                    strategy: Box::new(PipelineStrategy {
                        denoiser: Some("encoder".to_string()),
                        ..bare_strategy(PipelineStrategyKind::Iterative)
                    }),
                    run_on: None,
                }],
                ..bare_strategy(PipelineStrategyKind::Composite)
            },
            phases: BTreeMap::new(),
            vision: None,
        };
        let error = PipelinePlan::from_spec(&spec, &SchedulerRegistry::builtin()).unwrap_err();
        assert!(
            error.to_string().contains("iterative stage"),
            "unexpected error: {error}"
        );
    }

    fn vision_config(placeholder_id: i64, tpt: usize) -> PipelineVisionConfig {
        PipelineVisionConfig {
            image_placeholder_token_id: Some(placeholder_id),
            tokens_per_tile: Some(tpt),
        }
    }

    #[test]
    fn image_placeholder_expansion_replaces_tokens() {
        // [1, PLACEHOLDER, 2] with 2 tiles × 3 tokens/tile → [1, IMG, IMG, IMG, IMG, IMG, IMG, 2]
        let tokens: Vec<TokenId> = vec![1, 100, 2];
        let cfg = vision_config(100, 3);
        let expanded =
            expand_image_placeholders_count_based(tokens, Some(2), Some(&cfg)).unwrap();
        assert_eq!(expanded, vec![1, 100, 100, 100, 100, 100, 100, 2]);
    }

    #[test]
    fn image_placeholder_expansion_multiple_placeholders_errors() {
        // Count-based path only supports a single placeholder; >1 must error.
        let tokens: Vec<TokenId> = vec![100, 5, 100];
        let cfg = vision_config(100, 4);
        let err =
            expand_image_placeholders_count_based(tokens, Some(1), Some(&cfg)).unwrap_err();
        assert!(
            err.to_string().contains("multi-image count-based expansion is not supported"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn image_placeholder_expansion_none_tiles_is_noop() {
        let tokens: Vec<TokenId> = vec![1, 100, 2];
        let cfg = vision_config(100, 256);
        let result = expand_image_placeholders_count_based(tokens.clone(), None, Some(&cfg)).unwrap();
        assert_eq!(result, tokens);
    }

    #[test]
    fn image_placeholder_expansion_no_vision_config_with_tiles_errors() {
        let tokens: Vec<TokenId> = vec![1, 100, 2];
        let err =
            expand_image_placeholders_count_based(tokens, Some(1), None).unwrap_err();
        assert!(err.to_string().contains("no vision section"));
    }

    #[test]
    fn image_placeholder_expansion_incomplete_contract_errors() {
        let tokens: Vec<TokenId> = vec![1, 100, 2];
        let cfg = PipelineVisionConfig {
            image_placeholder_token_id: Some(100),
            tokens_per_tile: None,
        };
        let err =
            expand_image_placeholders_count_based(tokens, Some(1), Some(&cfg)).unwrap_err();
        assert!(err.to_string().contains("vision contract is incomplete"));
    }

    #[test]
    fn image_placeholder_expansion_missing_placeholder_errors() {
        let tokens: Vec<TokenId> = vec![1, 2, 3];
        let cfg = vision_config(100, 4);
        let err =
            expand_image_placeholders_count_based(tokens, Some(1), Some(&cfg)).unwrap_err();
        assert!(err.to_string().contains("no image placeholder token"));
    }

    #[test]
    fn image_placeholder_expansion_negative_id_errors() {
        let tokens: Vec<TokenId> = vec![1, 2];
        let cfg = PipelineVisionConfig {
            image_placeholder_token_id: Some(-1),
            tokens_per_tile: Some(4),
        };
        let err =
            expand_image_placeholders_count_based(tokens, Some(1), Some(&cfg)).unwrap_err();
        assert!(err.to_string().contains("out of range"));
    }

    #[test]
    fn image_placeholder_expansion_tokens_per_tile_zero_errors() {
        let tokens: Vec<TokenId> = vec![1, 100, 2];
        let cfg = vision_config(100, 0);
        let err =
            expand_image_placeholders_count_based(tokens, Some(1), Some(&cfg)).unwrap_err();
        assert!(
            err.to_string().contains("tokens_per_tile is 0"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn image_placeholder_expansion_zero_tiles_produces_empty_errors() {
        // tokens_per_tile=4, num_tiles=0 → expansion=0 → prompt becomes empty
        let tokens: Vec<TokenId> = vec![100];
        let cfg = vision_config(100, 4);
        let err =
            expand_image_placeholders_count_based(tokens, Some(0), Some(&cfg)).unwrap_err();
        assert!(
            err.to_string().contains("empty token sequence"),
            "unexpected error: {err}"
        );
    }
}

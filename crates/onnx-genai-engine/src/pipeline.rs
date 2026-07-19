//! Multi-model pipeline orchestrator.

use crate::decode::{
    DecodeState, clone_value, extract_next_token_logits, run_decode_step_with_extra,
};
use crate::decode_loop::{DecodeLoopBackend, DecodeLoopState, run_decode_loop};
use crate::engine::{Engine, EngineConfig, model_requires_native_backend};
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
}

impl PipelineGenerateRequest {
    pub fn new(request: GenerateRequest) -> Self {
        Self {
            request,
            inputs: HashMap::new(),
            num_image_tiles: None,
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
        if config.decode_backend == EngineDecodeBackend::Native {
            anyhow::bail!("native backend not supported for pipeline models");
        }
        if config.decode_backend == EngineDecodeBackend::Auto {
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
            PipelinePlan::SinglePass(sp) => (None, sp.model.clone()),
            PipelinePlan::Iterative(it) => (None, it.denoiser.clone()),
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
        self.run_prompt_phase_components(&ar.prompt_components, &mut tensors)?;
        let decoder_extras = self.decoder_extra_inputs(&ar.decoder, &tensors)?;

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
            context_tokens: prompt_tokens,
            prompt_len: 0,
            generated_count: 0,
        };
        backend.prompt_len = backend.context_tokens.len();
        let mut loop_state = DecodeLoopState::new(0, options.seed, options.top_logprobs);
        run_decode_loop(
            &mut backend,
            &mut loop_state,
            &options,
            &chain,
            tokenizer,
            None,
            callback,
        )
    }

    pub fn spec(&self) -> &PipelineSpec {
        &self.models.directory.spec
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
            PipelinePlan::Autoregressive(_) => anyhow::bail!(
                "run_pipeline() runs single-pass or iterative pipelines; use generate() for \
                 autoregressive text pipelines"
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
        // Classifier-free guidance scale (active only when set and != 1.0).
        let guidance = plan.guidance_scale.filter(|s| *s != 1.0);
        // `constants` holds external inputs + prompt-phase outputs and is NOT
        // mutated by the loop, so a denoiser whose output port shares a name
        // with a conditioning input cannot clobber that conditioning. Denoiser
        // outputs live in a separate `loop_state`, keyed by output port.
        let mut constants = request.inputs;
        self.run_prompt_phase_components(&plan.prompt_components, &mut constants)?;

        let denoiser = self
            .models
            .session(&plan.denoiser)
            .with_context(|| format!("pipeline denoiser '{}' was not loaded", plan.denoiser))?;

        // Precompute the CFG unconditional conditioning once: the caller may
        // supply `{denoiser}.{port}.uncond` (the empty-prompt embedding, to match
        // real SD CFG); otherwise the conditioning is zeroed as a fallback.
        let cfg_uncond: Option<(String, Value)> = if guidance.is_some() {
            let port = plan
                .cfg_conditioning_input
                .clone()
                .context("classifier-free guidance requires cfg_conditioning_input")?;
            let cond_endpoint = format!("{}.{}", plan.denoiser, port);
            let uncond_endpoint = format!("{}.{}.uncond", plan.denoiser, port);
            let value = if let Some(u) = constants.get(&uncond_endpoint) {
                clone_value(u)?
            } else {
                let cond = constants
                    .get(&cond_endpoint)
                    .or_else(|| {
                        plan.dataflow
                            .iter()
                            .find(|e| e.to == cond_endpoint)
                            .and_then(|e| constants.get(&e.from))
                    })
                    .with_context(|| format!("cfg conditioning '{cond_endpoint}' not found"))?;
                Value::from_slice_f32(&vec![0.0f32; cond.numel()], cond.shape())?
            };
            Some((port, value))
        } else {
            None
        };

        // `carried` holds the value to feed each loop-carried INPUT port next
        // step (keyed by input port); `last_outputs` holds the denoiser's raw
        // outputs from the final step (keyed by output port). Keeping them
        // separate from the immutable `constants` pool prevents an output whose
        // name collides with a conditioning input from clobbering it.
        let mut carried: HashMap<String, Value> = HashMap::new();
        let mut last_outputs: HashMap<String, Value> = HashMap::new();
        for step in 0..plan.num_steps {
            // Timestep/sigma for this step: explicit schedule when provided,
            // otherwise the 0-based step index.
            let timestep = plan
                .timesteps
                .as_ref()
                .map(|ts| ts[step])
                .unwrap_or(step as f32);

            // Raw (unscaled) loop-carried sample feeding each loop input this
            // step: the seed at step 0, otherwise the value carried from the
            // previous step. The scheduler's `step` consumes these raw samples.
            let mut raw_samples: HashMap<String, Value> = HashMap::new();
            for (_, in_port) in &plan.loop_edges {
                let raw = if step == 0 {
                    let endpoint = format!("{}.{}", plan.denoiser, in_port);
                    constants.get(&endpoint).with_context(|| {
                        format!("missing iterative pipeline seed '{endpoint}' at step 0")
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
            if let Some(scheduler) = &plan.scheduler {
                for (_, in_port) in &plan.loop_edges {
                    let raw = &raw_samples[in_port];
                    if let Some(scaled) = scheduler.scale_input(step, plan.num_steps, raw)? {
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
                if let Some((port, value)) = &cfg_uncond {
                    cfg_overrides.retain(|(p, _)| *p != port.as_str());
                    cfg_overrides.push((port.as_str(), value));
                }
                let uncond_out = self.run_denoiser_pass(
                    denoiser,
                    plan,
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
                    let cond_v = cond_value.to_vec_f32()?;
                    let uncond_v = uncond_value.to_vec_f32()?;
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
                let next = if let Some(scheduler) = &plan.scheduler {
                    let sample = raw_samples.get(in_port).with_context(|| {
                        format!("missing loop-carried sample for '{}.{in_port}'", plan.denoiser)
                    })?;
                    scheduler.step(step, plan.num_steps, sample, model_output)?
                } else {
                    clone_value(model_output)?
                };
                carried.insert(in_port.clone(), next);
            }
            last_outputs = out_map;
        }

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
        self.run_prompt_phase_components(&plan.final_components, &mut tensors)?;
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
                inputs.push((port.to_string(), clone_value(over_value)?));
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
                if step == 0 {
                    constants.get(&endpoint).with_context(|| {
                        format!("missing iterative pipeline seed '{endpoint}' at step 0")
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
            inputs.push((port.to_string(), clone_value(value)?));
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

    /// Run a single-pass pipeline: prompt-phase components once, then one
    /// forward invocation of the strategy `model`.
    fn run_single_pass(
        &self,
        request: PipelineGenerateRequest,
    ) -> anyhow::Result<PipelineTensors> {
        let PipelinePlan::SinglePass(plan) = &self.plan else {
            anyhow::bail!("internal error: run_single_pass on a non-single-pass plan");
        };
        let mut tensors = request.inputs;
        self.run_prompt_phase_components(&plan.prompt_components, &mut tensors)?;

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
            let outputs = session
                .run(&refs)
                .map_err(|e| anyhow::anyhow!("ORT pipeline component '{component}' failed: {e}"))?;
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
            inputs.push((info.name.clone(), clone_value(value)?));
        }
        Ok(inputs)
    }

    fn decoder_extra_inputs(
        &self,
        decoder: &str,
        tensors: &PipelineTensors,
    ) -> anyhow::Result<Vec<(String, Value)>> {
        let mut extras = Vec::new();
        for edge in self
            .plan
            .edges_to_component(decoder)
            .filter(|edge| endpoint_component(&edge.from).is_some_and(|from| from != decoder))
        {
            let (_, input) = parse_endpoint(&edge.to)?;
            let value = tensors
                .get(&edge.from)
                .with_context(|| format!("missing routed pipeline tensor '{}'", edge.from))?;
            extras.push((input.to_string(), clone_value(value)?));
        }
        Ok(extras)
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

struct PipelineDecodeLoopBackend<'a> {
    decoder: &'a Session,
    decoder_state: &'a mut DecodeState,
    decoder_extras: &'a [(String, Value)],
    context_tokens: Vec<TokenId>,
    prompt_len: usize,
    generated_count: usize,
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
        let outputs = run_decode_step_with_extra(
            self.decoder,
            self.decoder_state,
            &input_tokens,
            past_len,
            self.decoder_extras,
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
    SinglePass(SinglePassPlan),
    Iterative(IterativePlan),
}

/// Token-by-token decoder pipeline (optionally with prompt-phase encoders).
#[derive(Debug, Clone)]
struct AutoregressivePlan {
    decoder: String,
    prompt_components: Vec<String>,
    dataflow: Vec<DataflowEdge>,
}

/// One forward invocation of a single model with no runtime-managed loop.
#[derive(Debug, Clone)]
struct SinglePassPlan {
    model: String,
    /// Components that run once before the model (e.g. an encoder).
    prompt_components: Vec<String>,
    dataflow: Vec<DataflowEdge>,
}

/// Bounded iterative loop (diffusion denoise / other fixed-step refinement).
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
                    cfg.use_karras_sigmas.unwrap_or(false),
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
                    cfg.use_karras_sigmas.unwrap_or(false),
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
                Ok(Arc::new(MaskedDiffusion { mask_token_id }) as Arc<dyn Scheduler>)
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

/// Masked (discrete) language diffusion: the loop-carried tensor is an int64
/// token sequence, the denoiser emits `[B, S, V]` logits, and each step commits
/// the highest-confidence still-masked positions (argmax), unmasking them
/// progressively so all masked positions are filled by the final step.
#[derive(Debug, Clone)]
struct MaskedDiffusion {
    mask_token_id: i64,
}

impl Scheduler for MaskedDiffusion {
    fn step(
        &self,
        step: usize,
        num_steps: usize,
        tokens: &Value,
        logits: &Value,
    ) -> anyhow::Result<Value> {
        let token_shape = tokens.shape().to_vec();
        let toks = tokens.to_vec_i64()?;
        let n = toks.len();
        let logit_shape = logits.shape();
        let vocab = *logit_shape
            .last()
            .context("masked_diffusion logits must be rank >= 1")? as usize;
        if vocab == 0 || n == 0 || logits.numel() != n * vocab {
            anyhow::bail!(
                "masked_diffusion shape mismatch: tokens {token_shape:?}, logits {logit_shape:?}"
            );
        }
        let lg = logits.to_vec_f32()?;
        // Per-position argmax + confidence (max logit).
        let mut pred = vec![0i64; n];
        let mut conf = vec![f32::MIN; n];
        for i in 0..n {
            let row = &lg[i * vocab..(i + 1) * vocab];
            let mut best = (0usize, f32::MIN);
            for (j, &x) in row.iter().enumerate() {
                if x > best.1 {
                    best = (j, x);
                }
            }
            pred[i] = best.0 as i64;
            conf[i] = best.1;
        }
        // Commit the highest-confidence still-masked positions this step.
        let mut masked: Vec<usize> = (0..n).filter(|&i| toks[i] == self.mask_token_id).collect();
        let mut out = toks.clone();
        if !masked.is_empty() {
            masked.sort_by(|&a, &b| {
                conf[b].partial_cmp(&conf[a]).unwrap_or(std::cmp::Ordering::Equal)
            });
            let remaining_steps = num_steps.saturating_sub(step).max(1);
            let commit = masked.len().div_ceil(remaining_steps);
            for &i in masked.iter().take(commit) {
                out[i] = pred[i];
            }
        }
        Value::from_slice_i64(&out, &token_shape).map_err(Into::into)
    }
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
        for k in 0..num_steps {
            let t = ascending[num_steps - 1 - k];
            let a_t = alpha_cumprod[t];
            let a_prev = if k + 1 < num_steps {
                alpha_cumprod[ascending[num_steps - 1 - (k + 1)]]
            } else {
                1.0
            };
            steps.push((a_t, a_prev));
        }
        Ok(Self { steps })
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
            DdimSchedule::step(self, step, &sample.to_vec_f32()?, &model_output.to_vec_f32()?)?;
        Value::from_slice_f32(&stepped, &shape).map_err(Into::into)
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
}

impl EulerSchedule {
    fn with_schedule(
        num_train_timesteps: usize,
        beta_start: f32,
        beta_end: f32,
        beta_schedule: &str,
        num_steps: usize,
        use_karras: bool,
    ) -> anyhow::Result<Self> {
        if num_train_timesteps < 2 {
            anyhow::bail!("scheduler num_train_timesteps must be >= 2");
        }
        if num_steps == 0 || num_steps > num_train_timesteps {
            anyhow::bail!(
                "scheduler num_steps ({num_steps}) must be in 1..={num_train_timesteps}"
            );
        }
        if use_karras {
            let sigmas =
                karras_sigmas(num_train_timesteps, beta_start, beta_end, beta_schedule, num_steps)?;
            return Ok(Self { sigmas });
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
        for k in 0..num_steps {
            let idx = num_steps - 1 - k;
            let t = idx as f32 * denom / ts_denom;
            sigmas.push(interp(t));
        }
        sigmas.push(0.0);
        Ok(Self { sigmas })
    }

    /// `init_noise_sigma` — the factor the caller must apply to the initial
    /// random latent so it lives in the scheduler's sigma space.
    #[allow(dead_code)]
    fn init_noise_sigma(&self) -> f32 {
        self.sigmas[0]
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
        let stepped = self.step_vec(step, &sample.to_vec_f32()?, &model_output.to_vec_f32()?)?;
        Value::from_slice_f32(&stepped, &shape).map_err(Into::into)
    }

    fn scale_input(
        &self,
        step: usize,
        _num_steps: usize,
        sample: &Value,
    ) -> anyhow::Result<Option<Value>> {
        let shape = sample.shape().to_vec();
        let scaled = self.scale(step, &sample.to_vec_f32()?);
        Ok(Some(Value::from_slice_f32(&scaled, &shape)?))
    }
}

/// DPM-Solver++ (2M) — a fast *multistep* deterministic scheduler and the default
/// sampler in most Stable Diffusion / ComfyUI workflows. Order-2 in log-SNR (λ)
/// space using the previous step's data prediction (`x0`), with a first-order step
/// at the start and (for <15 steps) a first-order final step. Matches diffusers
/// `DPMSolverMultistepScheduler(algorithm_type="dpmsolver++", solver_type="midpoint")`.
/// Unlike Euler it does NOT scale the model input (`scale_model_input` is identity)
/// and its `init_noise_sigma` is 1.0 (the seed is unscaled).
#[derive(Debug)]
struct Dpmpp2m {
    /// Inference sigmas, descending, with a trailing `0.0`. Length `num_steps + 1`.
    sigmas: Vec<f32>,
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
        use_karras: bool,
    ) -> anyhow::Result<Self> {
        if num_train_timesteps < 2 {
            anyhow::bail!("scheduler num_train_timesteps must be >= 2");
        }
        if num_steps == 0 || num_steps > num_train_timesteps {
            anyhow::bail!(
                "scheduler num_steps ({num_steps}) must be in 1..={num_train_timesteps}"
            );
        }
        if use_karras {
            let sigmas =
                karras_sigmas(num_train_timesteps, beta_start, beta_end, beta_schedule, num_steps)?;
            return Ok(Self {
                sigmas,
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
        let mut sigmas: Vec<f32> = ts_int
            .iter()
            .map(|&t| train[t.min(num_train_timesteps - 1)])
            .collect();
        sigmas.push(0.0);
        Ok(Self {
            sigmas,
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

impl Scheduler for Dpmpp2m {
    fn step(
        &self,
        step: usize,
        num_steps: usize,
        sample: &Value,
        model_output: &Value,
    ) -> anyhow::Result<Value> {
        let shape = sample.shape().to_vec();
        let x = sample.to_vec_f32()?;
        let eps = model_output.to_vec_f32()?;
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
        if step == 0 {
            *prev = None; // reset at the start of each denoise loop
        }
        let lower_order_final = step + 1 == num_steps && num_steps < 15;
        let first_order = step == 0 || lower_order_final || prev.is_none();

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
}

impl PipelinePlan {
    fn from_spec(spec: &PipelineSpec, schedulers: &SchedulerRegistry) -> anyhow::Result<Self> {
        // A composite whose stages contain an autoregressive decoder is treated
        // as an autoregressive text pipeline (unchanged legacy behavior). Pure
        // iterative / single-pass composites are a follow-up.
        if let Some(decoder) = autoregressive_decoder(&spec.strategy) {
            return Self::autoregressive(spec, decoder);
        }
        match spec.strategy.kind {
            PipelineStrategyKind::SinglePass => Self::single_pass(spec),
            PipelineStrategyKind::Iterative => Self::iterative(spec, schedulers),
            PipelineStrategyKind::Composite => anyhow::bail!(
                "composite pipeline strategy without an autoregressive decoder is not yet \
                 supported"
            ),
            PipelineStrategyKind::Autoregressive => {
                anyhow::bail!("autoregressive strategy is missing its 'decoder' component")
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
        Ok(Self::Autoregressive(AutoregressivePlan {
            decoder,
            prompt_components,
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

        // Classifier-free guidance requires a declared conditioning input to
        // zero on the unconditional pass.
        let guidance_active = spec.strategy.guidance_scale.is_some_and(|s| s != 1.0);
        if guidance_active && spec.strategy.cfg_conditioning_input.is_none() {
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

        Ok(Self::Iterative(IterativePlan {
            denoiser,
            num_steps,
            guidance_scale: spec.strategy.guidance_scale,
            prompt_components,
            final_components,
            loop_edges,
            timestep_input: spec.strategy.timestep_input.clone(),
            timesteps: spec.strategy.timesteps.clone(),
            scheduler: build_scheduler(spec.strategy.scheduler_config.as_ref(), num_steps, schedulers)?,
            cfg_conditioning_input: spec.strategy.cfg_conditioning_input.clone(),
            dataflow: spec.dataflow.clone(),
        }))
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
            Self::SinglePass(plan) => &plan.dataflow,
            Self::Iterative(plan) => &plan.dataflow,
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
        | PipelineStrategyKind::Other(_) => None,
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
                scheduler_config: None,
                cfg_conditioning_input: None,
                guidance_scale: None,
                state: None,
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
                            scheduler_config: None,
                            cfg_conditioning_input: None,
                            guidance_scale: None,
                            state: None,
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
                            scheduler_config: None,
                            cfg_conditioning_input: None,
                            guidance_scale: None,
                            state: None,
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

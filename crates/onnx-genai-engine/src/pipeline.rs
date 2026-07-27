//! Multi-model pipeline orchestrator.

use crate::decode::{
    DecodeState, apply_paged_sliding_window, clone_value, extract_next_token_logits_with_io,
    is_present_output, run_decode_step_with_extra,
};
use crate::decode_loop::{DecodeLoopBackend, DecodeLoopState, run_decode_loop};
use crate::engine::{
    Engine, EngineConfig, model_requires_native_backend, requested_decode_backend,
    resolved_host_ram_budget,
};
use crate::kv_bridge::{
    KvModelInfo, attach_pages_to_sequence, infer_kv_model_info, load_materialized_past,
    mirror_present_kv_to_pages, sequence_pages_for_len,
};
use crate::logits::TokenId;
use crate::pipeline_cache::{
    ComponentOutputCache, Digest, DigestBuilder, PREFIX_KEY_PREAMBLE, PipelineCacheStats,
    RetainedContext, absorb_value, digest_named_values, graph_is_deterministic, prefix_key,
};
use crate::processors::build_processor_chain;
use crate::{
    EngineDecodeBackend, GeneratePrompt, GenerateRequest, GenerateResult, GenerateTokenCallback,
};
use anyhow::Context;
use onnx_genai_kv::{PagedKvCache, PrefixCache, SequenceId};
use onnx_genai_metadata::{
    AbsentInputKind, DataflowEdge, PhaseRunOn, PipelineSpec, PipelineStrategy,
    PipelineStrategyKind, PipelineVisionConfig, SchedulerSpec, TensorDimension,
};
use onnx_genai_ort::{
    DataType, PipelineModelDirectory, PipelineModels, Session, SessionOptions, Tokenizer, Value,
};
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::sync::{Arc, Mutex};

mod schedulers;
pub use schedulers::{Scheduler, SchedulerFactory, SchedulerRegistry};
use schedulers::{
    PredictionType, dpm_alpha_sigma, epsilon_from_model_output, sigma_to_t, spacing_sigmas,
    training_sigmas, x0_from_model_output,
};

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
    /// Opaque metadata-declared presence keys active for this request.
    ///
    /// Empty preserves the historical behavior for pipelines without optional
    /// inputs or presence-gated components.
    pub present: BTreeSet<String>,
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
            present: BTreeSet::new(),
            num_image_tiles: None,
            iterative_overrides: IterativeOverrides::default(),
        }
    }

    pub fn with_input(mut self, endpoint: impl Into<String>, value: Value) -> Self {
        self.inputs.insert(endpoint.into(), value);
        self
    }

    pub fn with_presence(mut self, key: impl Into<String>) -> Self {
        self.present.insert(key.into());
        self
    }

    pub fn with_present_keys(mut self, keys: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.present.extend(keys.into_iter().map(Into::into));
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
    fixed_state_budget_bytes: u64,
    /// Memoized prompt-phase component outputs, so a repeated attachment does
    /// not re-run its encoder. Behind a `RefCell` because the prompt phase runs
    /// from `&self` paths (single-pass, iterative) as well as `&mut self` ones.
    component_cache: RefCell<ComponentOutputCache>,
    /// Components whose graphs contain only deterministic operators, and whose
    /// outputs may therefore be memoized. Computed once at load.
    memoizable_components: BTreeSet<String>,
    /// Decoder KV left over from the previous generation, and the identity of
    /// the prompt and attachments that produced it.
    ///
    /// Only used when the decoder's KV cannot be paged; the paged cache below
    /// supersedes it, because it holds many prefixes instead of one.
    retained: Option<RetainedContext>,
    /// Paged KV for the decoder, when its `present.*` outputs describe a layout
    /// the page table can address.
    paged: Option<PipelinePagedKv>,
}

/// Paged KV storage for an autoregressive pipeline decoder.
///
/// This is the same machinery the single-model engine uses, which is the point:
/// pages are reference-counted, so several conversations that open with the same
/// system prompt hold *one* copy of its KV between them rather than one each.
/// A single retained context can only ever serve whoever spoke last.
struct PipelinePagedKv {
    kv_model: KvModelInfo,
    cache: PagedKvCache,
    /// Sequence claimed for the generation in flight.
    ///
    /// A generation can fail at any `?` between claiming a sequence and
    /// publishing it, and an abandoned sequence keeps its pages referenced
    /// forever. Recording it here means the next admission — or the explicit
    /// discard on the error path — can always find and free it.
    active: Option<SequenceId>,
    /// Radix trie over prefix keys. The keys are not bare prompts: each carries
    /// a digest of the request's attachments ahead of its tokens, because image
    /// expansion makes different pictures produce identical token sequences and
    /// a bare-token key would hand one image's KV to another.
    prefix: PrefixCache,
}

/// The concrete backend a pipeline runs on. A pipeline never mixes backends:
/// every component is instantiated through the same backend's
/// [`ComponentSession`](onnx_genai_metadata::ComponentSession) implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PipelineBackend {
    Ort,
    Native,
}

/// Resolve an `Auto` decode-backend request to a concrete [`PipelineBackend`].
/// Native is selected only when some component declares operators that require
/// the native backend; otherwise ORT.
fn resolve_auto_pipeline_backend(
    directory: &PipelineModelDirectory,
) -> anyhow::Result<PipelineBackend> {
    for model_path in directory.model_paths.values() {
        if model_requires_native_backend(model_path)? {
            return Ok(PipelineBackend::Native);
        }
    }
    Ok(PipelineBackend::Ort)
}

/// Native backend unavailable at build time: the crate was compiled without the
/// `native-backend` feature, so no native component sessions can be constructed.
#[cfg(not(feature = "native-backend"))]
fn native_backend_not_compiled_error() -> anyhow::Error {
    anyhow::anyhow!(
        "the native backend was requested for a pipeline model, but this build of \
         onnx-genai-engine was compiled without the 'native-backend' feature. Rebuild with \
         `--features native-backend` (and `cuda` for GPU) to run pipelines natively, or set \
         decode_backend = EngineDecodeBackend::Ort (or ONNX_GENAI_BACKEND=ort) to use ONNX \
         Runtime."
    )
}

/// Construct all pipeline components through the native
/// [`ComponentSession`](onnx_genai_metadata::ComponentSession) seam, then report
/// the first genuinely-unimplemented step for native pipeline decode.
///
/// Backend-neutral construction (GAP 1) is complete once every component loads
/// and exposes graph I/O metadata through the trait. Wiring those neutral
/// sessions into the pipeline decode loop — which still routes per-step state
/// through ORT `Value`/`Session` — is the remaining native work (GAP 3), so
/// this returns a clear, actionable error naming that precise next blocker
/// rather than a blanket "native backend not supported" rejection at
/// construction.
#[cfg(feature = "native-backend")]
fn build_native_pipeline_and_report_gap(
    directory: &PipelineModelDirectory,
    config: &EngineConfig,
) -> anyhow::Error {
    let components = match build_native_pipeline_components(directory, config) {
        Ok(components) => components,
        Err(err) => return err,
    };
    let component_list = components.keys().cloned().collect::<Vec<_>>().join(", ");
    anyhow::anyhow!(
        "native pipeline decode is not yet implemented. All {} pipeline component(s) loaded \
         successfully on the native backend and expose their graph I/O through the \
         backend-neutral component-session interface (components: {}), so backend selection and \
         construction are backend-neutral. Native target decode now accepts metadata-declared \
         token or embedding sequence inputs plus arbitrary named routed step tensors. The \
         remaining GAP 3 work is replacing `DecodeState` and `PipelineDecodeLoopBackend` ownership \
         of ORT `Value`/`Session` with backend-neutral tensors/component sessions, then invoking \
         the native target step with those routed tensors. To run this pipeline today, set \
         decode_backend = EngineDecodeBackend::Ort (or \
         ONNX_GENAI_BACKEND=ort).",
        components.len(),
        component_list,
    )
}

/// Load every declared pipeline component on the native backend, exposing each
/// through the backend-neutral [`ComponentSession`] seam. Returns the components
/// keyed by name so backend selection instantiates them all consistently.
#[cfg(feature = "native-backend")]
fn build_native_pipeline_components(
    directory: &PipelineModelDirectory,
    config: &EngineConfig,
) -> anyhow::Result<
    std::collections::BTreeMap<String, Box<dyn onnx_genai_metadata::ComponentSession>>,
> {
    use crate::native_component::NativeComponentSession;
    use onnx_genai_metadata::ComponentSession;

    let device = crate::engine::resolve_native_decode_device(
        config.native_device,
        &SessionOptions::default(),
    )?;
    let mut components: std::collections::BTreeMap<String, Box<dyn ComponentSession>> =
        std::collections::BTreeMap::new();
    for (name, path) in &directory.model_paths {
        let session = NativeComponentSession::load(path, device).with_context(|| {
            format!("failed to construct pipeline component '{name}' on the native backend")
        })?;
        components.insert(name.clone(), Box::new(session));
    }
    Ok(components)
}

/// Components whose graphs contain only deterministic operators.
///
/// Read once at load, because a component's declared phase says when it runs,
/// never that it is pure — a graph with `RandomNormal` in it would otherwise be
/// memoized and hand back the same draw forever. A model that cannot be read or
/// parsed is simply left out: declining to cache is always safe, and refusing to
/// load a pipeline over a cache optimization would not be.
fn deterministic_components(directory: &PipelineModelDirectory) -> BTreeSet<String> {
    directory
        .model_paths
        .iter()
        .filter(|(_, path)| {
            onnx_runtime_loader::read_model_binary(path)
                .ok()
                .and_then(|bytes| onnx_runtime_loader::proto::decode_model(&bytes).ok())
                .is_some_and(|model| graph_is_deterministic(&model))
        })
        .map(|(name, _)| name.clone())
        .collect()
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

    /// Load a pipeline with explicit session options, chiefly to pin the
    /// execution provider.
    ///
    /// [`SessionOptions::default`] resolves the provider from the process
    /// environment, which is read once and cached; a caller that wants to choose
    /// a provider *after* startup — an interactive session switching devices —
    /// has to pass one in instead.
    pub fn from_dir_with_session_options(
        pipeline_dir: &Path,
        config: EngineConfig,
        session_options: SessionOptions,
    ) -> anyhow::Result<Self> {
        Self::build(
            pipeline_dir,
            config,
            &SchedulerRegistry::builtin(),
            session_options,
        )
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
        Self::build(pipeline_dir, config, schedulers, SessionOptions::default())
    }

    fn build(
        pipeline_dir: &Path,
        config: EngineConfig,
        schedulers: &SchedulerRegistry,
        session_options: SessionOptions,
    ) -> anyhow::Result<Self> {
        let decode_backend = requested_decode_backend(config.decode_backend)?;
        // Select ONE backend for the whole pipeline (never a mix). Explicit
        // backends resolve without touching the model directory (so a bad
        // request fails fast); `Auto` inspects the components' declared
        // operators, selecting native only when some component requires it.
        let backend = match decode_backend {
            EngineDecodeBackend::Ort => PipelineBackend::Ort,
            EngineDecodeBackend::Native => PipelineBackend::Native,
            EngineDecodeBackend::Auto => {
                let directory = PipelineModelDirectory::load(pipeline_dir)
                    .map_err(|e| anyhow::anyhow!("Failed to resolve pipeline models: {}", e))?;
                resolve_auto_pipeline_backend(&directory)?
            }
        };
        if backend == PipelineBackend::Native {
            // The native backend constructs every declared component through the
            // backend-neutral `ComponentSession` seam (GAP 1); no ORT type
            // reaches this path. When the crate is built without the
            // `native-backend` feature there is nothing to construct.
            #[cfg(not(feature = "native-backend"))]
            {
                return Err(native_backend_not_compiled_error());
            }
            #[cfg(feature = "native-backend")]
            {
                let directory = PipelineModelDirectory::load(pipeline_dir)
                    .map_err(|e| anyhow::anyhow!("Failed to resolve pipeline models: {}", e))?;
                return Err(build_native_pipeline_and_report_gap(&directory, &config));
            }
        }
        let models = PipelineModels::load_with_options(pipeline_dir, session_options)
            .map_err(|e| anyhow::anyhow!("Failed to load pipeline models: {}", e))?;
        let plan = PipelinePlan::from_spec(&models.directory.spec, schedulers)?;
        let memoizable_components = deterministic_components(&models.directory);
        // Only autoregressive pipelines drive a token-by-token decode loop and
        // therefore need a `DecodeState` + KV model info. Single-pass and
        // iterative (diffusion) pipelines run tensors through `run_pipeline`.
        let mut paged: Option<PipelinePagedKv> = None;
        let (decoder_state, tokenizer_component, fixed_state_budget_bytes) = match &plan {
            PipelinePlan::Autoregressive(ar) => {
                let decoder = models
                    .session(&ar.decoder)
                    .with_context(|| format!("pipeline decoder '{}' was not loaded", ar.decoder))?;
                let kv_model =
                    infer_kv_model_info(decoder, config.page_size, config.kv_cache_dtype)?;
                let fixed_state_budget_bytes =
                    resolved_host_ram_budget(&config, kv_model.as_ref())?;
                // A zero page size makes `div_ceil` panic and the page-boundary
                // walk below produce zeros forever, so it is refused rather than
                // carried into arithmetic that assumes it is positive.
                paged = kv_model
                    .filter(|kv_model| kv_model.tensor_config.page_size > 0)
                    .map(|kv_model| PipelinePagedKv {
                        cache: PagedKvCache::new_with_layer_tensor_configs(
                            kv_model.tensor_config.page_size,
                            kv_model.tensor_config.dtype,
                            kv_model.layer_configs.clone(),
                            config.num_gpu_pages,
                        ),
                        kv_model,
                        prefix: PrefixCache::new(),
                        active: None,
                    });
                let decoder_io = models
                    .directory
                    .spec
                    .models
                    .get(&ar.decoder)
                    .and_then(|component| component.io.as_ref());
                let positions = models.directory.spec.positions.as_ref();
                (
                    Some(DecodeState::new_with_io_positions_and_state_budget(
                        decoder,
                        decoder_io,
                        positions,
                        fixed_state_budget_bytes,
                    )?),
                    ar.decoder.clone(),
                    fixed_state_budget_bytes,
                )
            }
            // A nested-AR (multi-decoder TTS) pipeline drives its own outer/inner
            // decode loops with per-loop `DecodeState`s built inside the driver,
            // so no shared decode state is created here. The tokenizer component
            // is the outer decoder (talker).
            PipelinePlan::NestedAutoregressive(nested) => (None, nested.outer.clone(), 0),
            PipelinePlan::SinglePass(sp) => (None, sp.model.clone(), 0),
            PipelinePlan::Iterative(it) => (None, it.denoiser.clone(), 0),
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
                0,
            ),
        };
        Ok(Self {
            models,
            plan,
            decoder_state,
            tokenizer_component,
            fixed_state_budget_bytes,
            component_cache: RefCell::new(ComponentOutputCache::new(
                usize::try_from(config.pipeline_cache_bytes).unwrap_or(usize::MAX),
            )),
            memoizable_components,
            retained: None,
            paged,
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
        let present = pipeline_request.present.clone();
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
            &present,
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
        let present = pipeline_request.present.clone();
        self.ensure_component_present(&ar.decoder, &present, "autoregressive decoder")?;

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

        // Decide how much of the previous turn's decoder KV this prompt can
        // keep before anything is rebuilt, because the answer decides whether
        // the decode state is recreated or carried over.
        let inputs_digest = Self::digest_request_identity(&pipeline_request);
        // The paged cache supersedes the single retained context wherever it is
        // available: it holds many prefixes rather than only the last one.
        let paged_enabled = self.paged.is_some() && inputs_digest.is_some();
        let reused = if paged_enabled {
            0
        } else {
            self.reusable_prefix_len(inputs_digest, &prompt_tokens)
        };
        // Any failure below leaves the decoder KV in an unknown state, so the
        // retention is dropped now and only re-established on success.
        self.retained = None;

        let mut tensors = self.prepare_request_tensors(pipeline_request.inputs, &present)?;
        // Seed the prompt token ids into the shared pool so a prompt-phase
        // component that consumes `input_ids` (e.g. a text encoder) can run.
        self.seed_prompt_token_inputs(&ar.prompt_components, &prompt_tokens, &mut tensors)?;
        self.run_prompt_phase_components(
            &ar.prompt_components,
            &mut tensors,
            "prologue",
            &present,
            None,
        )?;

        // Static routing from prompt-phase and per-step producers into the
        // decoder. Every non-self edge into the decoder is recomputed from the
        // shared pool on each step, so `every_step` outputs are always fresh and
        // `prompt_only` conditioning stays cached (it is simply re-read).
        let decoder_in_edges = self.decoder_in_edges(&ar.decoder, &present, &tensors)?;
        // Owned per-step component bindings (paired with their sessions below).
        // Built before `decoder_state` is taken mutably so the immutable borrow
        // used to enumerate graph ports is released first.
        let step_bindings = self.build_step_bindings(&ar.step_components, &present)?;

        // A decoder whose position ids arrive over a dataflow edge receives one
        // tensor covering the whole prompt. Prefilling only a suffix would hand
        // it positions for tokens it is not being given, so such a pipeline
        // recomputes rather than reuses.
        let reused = if reused > 0 && self.decoder_positions_are_routed(&decoder_in_edges) {
            0
        } else {
            reused
        };

        let positions_routed = self.decoder_positions_are_routed(&decoder_in_edges);
        // A paged sequence starts from a fresh decode state, because the shared
        // prefix is loaded into it wholesale rather than carried over.
        let mut paged_session = None;
        let mut reused = reused;
        if paged_enabled && !positions_routed {
            self.decoder_state = Some(Self::new_decoder_state(
                &self.models,
                &ar.decoder,
                self.fixed_state_budget_bytes,
            )?);
            let inputs = inputs_digest.expect("paged_enabled implies a digest");
            let decoder = self
                .models
                .session(&ar.decoder)
                .with_context(|| format!("pipeline decoder '{}' was not loaded", ar.decoder))?;
            let paged = self.paged.as_mut().expect("paged_enabled implies storage");
            let state = self
                .decoder_state
                .as_mut()
                .expect("the decode state was just built");
            let (seq, shared) =
                Self::admit_paged_sequence(paged, state, decoder, inputs, &prompt_tokens)?;
            paged_session = Some((seq, inputs));
            reused = shared;
        }

        let chain = build_processor_chain(&options, Some(self.tokenizer()?))?;
        if reused == 0 && paged_session.is_none() {
            self.decoder_state = Some(Self::new_decoder_state(
                &self.models,
                &ar.decoder,
                self.fixed_state_budget_bytes,
            )?);
        }

        // Encoder-decoder pipelines bind the encoder's static cross-attention KV
        // to the decoder every step. Resolve it once here, after the prompt-phase
        // encoder prologue has published its `present_*_cross_%d` outputs into the
        // shared pool; the tensors are invariant across the decode loop.
        let cross_kv_pairs = self
            .decoder_state
            .as_ref()
            .expect("autoregressive pipeline has decode state")
            .io
            .cross_kv_pairs
            .clone();
        let static_cross_kv = self.static_cross_kv_bindings(&cross_kv_pairs, &tensors)?;

        let decoder = self
            .models
            .session(&ar.decoder)
            .with_context(|| format!("pipeline decoder '{}' was not loaded", ar.decoder))?;
        // Pair every `every_step` binding with its loaded session. This is the
        // generic replacement for the old one-output `inputs_embeds` fusion.
        let step_components = step_bindings
            .into_iter()
            .map(|binding| {
                let session = self.models.session(&binding.component).with_context(|| {
                    format!(
                        "pipeline every_step component '{}' was not loaded",
                        binding.component
                    )
                })?;
                Ok((binding, session))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let tokenizer = self
            .models
            .tokenizer_for(&self.tokenizer_component)
            .with_context(|| {
                format!("no tokenizer available for '{}'", self.tokenizer_component)
            })?;
        let paged_mirror = match (paged_session, self.paged.as_mut()) {
            (Some((seq, _)), Some(paged)) => Some(PagedMirror {
                mirrored_tokens: 0,
                exhausted: false,
                windowed: false,
                kv_model: &paged.kv_model,
                cache: &mut paged.cache,
                seq,
            }),
            _ => None,
        };
        let mut backend = PipelineDecodeLoopBackend {
            decoder,
            decoder_state: self
                .decoder_state
                .as_mut()
                .expect("autoregressive pipeline has decode state"),
            paged: paged_mirror,
            pool: &mut tensors,
            step_components,
            decoder_in_edges,
            static_cross_kv,
            context_tokens: prompt_tokens,
            retained_len: reused,
            prompt_len: 0,
            generated_count: 0,
            kv_len: reused,
        };
        // Prefill only what the retained KV does not already cover.
        backend.prompt_len = backend.context_tokens.len() - backend.retained_len;
        let prefilled = backend.prompt_len;
        let mut loop_state = DecodeLoopState::new(reused, options.seed, options.top_logprobs);
        // Taken without `?` so a failed generation still releases its sequence
        // below: an abandoned sequence holds its pages out of the pool for the
        // life of the process.
        let result = run_decode_loop(
            &mut backend,
            &mut loop_state,
            &options,
            &chain,
            tokenizer,
            None,
            callback,
        );
        // Exactly the tokens whose KV the decoder now holds. Truncated to
        // `kv_len` rather than taken whole: the last sampled token was committed
        // to the context but never fed to the decoder, so its KV does not exist
        // and the next turn must prefill it.
        let mut final_context = backend.context_tokens.clone();
        final_context.truncate(backend.kv_len);
        let retains_kv = backend.decoder_state.use_kv;
        // How far mirroring actually got. Equal to the context length unless
        // the page pool ran dry, in which case only this prefix may be
        // published for reuse.
        let mirrored_tokens = backend.paged.as_ref().map_or(0, |mirror| {
            if mirror.windowed {
                0
            } else {
                mirror.mirrored_tokens
            }
        });
        let result = match result {
            Ok(result) => result,
            Err(error) => {
                if let Some(paged) = self.paged.as_mut() {
                    paged.discard_active();
                }
                return Err(error);
            }
        };

        self.component_cache
            .borrow_mut()
            .note_prefix_reuse(reused, prefilled);
        match paged_session {
            // Publish what this generation computed so the next request can
            // attach to it, then let go of the sequence.
            Some((seq, inputs)) => {
                self.retire_paged_sequence(seq, inputs, &final_context, mirrored_tokens)?
            }
            None => {
                if retains_kv && let Some(inputs) = inputs_digest {
                    self.retained = Some(RetainedContext {
                        inputs,
                        tokens: final_context,
                    });
                }
            }
        }
        Ok((result, tensors))
    }

    /// A fresh decode state for `decoder`.
    fn new_decoder_state(
        models: &PipelineModels,
        decoder: &str,
        fixed_state_budget_bytes: u64,
    ) -> anyhow::Result<DecodeState> {
        let session = models
            .session(decoder)
            .with_context(|| format!("pipeline decoder '{decoder}' was not loaded"))?;
        let decoder_io = models
            .directory
            .spec
            .models
            .get(decoder)
            .and_then(|component| component.io.as_ref());
        DecodeState::new_with_io_positions_and_state_budget(
            session,
            decoder_io,
            models.directory.spec.positions.as_ref(),
            fixed_state_budget_bytes,
        )
    }

    /// Claim a paged sequence for this request, seeded with whatever cached
    /// prefix its attachments and tokens already share with earlier requests.
    ///
    /// Returns the sequence id and how many leading tokens the sequence already
    /// holds KV for. The reuse always stops at least one token short of the
    /// prompt, since a decode step needs an input to produce logits from.
    fn admit_paged_sequence(
        paged: &mut PipelinePagedKv,
        state: &mut DecodeState,
        decoder: &Session,
        inputs: Digest,
        prompt_tokens: &[TokenId],
    ) -> anyhow::Result<(SequenceId, usize)> {
        // Free anything a previous generation abandoned, then make room for this
        // one, before claiming any pages.
        paged.discard_active();
        paged.evict_until_free(
            prompt_tokens
                .len()
                .div_ceil(paged.cache.page_table.page_size),
        );
        let key = prefix_key(inputs, prompt_tokens);
        let seq = paged.cache.create_sequence();
        paged.active = Some(seq);
        let matched = paged
            .prefix
            .lookup_shared(&key, &mut paged.cache.page_table);

        // A match shorter than the preamble cannot happen — every stored key
        // begins with one — but treat it as no match rather than trusting it.
        let reusable = matched
            .matched_tokens
            .saturating_sub(PREFIX_KEY_PREAMBLE)
            .min(prompt_tokens.len().saturating_sub(1));
        if matched.matched_tokens > 0 {
            let pages = matched
                .page_ids
                .iter()
                .copied()
                .take(reusable.div_ceil(paged.cache.page_table.page_size))
                .collect::<Vec<_>>();
            for &page_id in &pages {
                paged.cache.page_table.retain(page_id);
            }
            paged
                .prefix
                .release_shared(&key, matched.matched_tokens, &mut paged.cache.page_table);
            if reusable > 0 {
                attach_pages_to_sequence(&mut paged.cache, seq, &pages, reusable)?;
                let materialized = paged
                    .cache
                    .materialize_sequence(seq)
                    .map_err(|e| anyhow::anyhow!("failed to materialize the shared prefix: {e}"))?;
                load_materialized_past(decoder, &paged.kv_model, state, &materialized)?;
                return Ok((seq, reusable));
            }
        }
        Ok((seq, 0))
    }

    /// Record this generation's KV under its prefix key and release the
    /// sequence.
    ///
    /// Pages the prefix cache kept are retained by it, so freeing the sequence
    /// returns only what nothing else refers to.
    fn retire_paged_sequence(
        &mut self,
        seq: SequenceId,
        inputs: Digest,
        tokens: &[TokenId],
        mirrored_tokens: usize,
    ) -> anyhow::Result<()> {
        let Some(paged) = self.paged.as_mut() else {
            return Ok(());
        };
        // Never publish past what was mirrored. If the pool ran dry mid-decode
        // the later pages were never written, and a key covering them would
        // hand a future request KV that does not exist.
        let tokens = &tokens[..tokens.len().min(mirrored_tokens)];
        // Publish at every page boundary, not only at the full length.
        //
        // The trie only reports a match where something was inserted, so a
        // prompt that diverges from this one matches nothing unless the shared
        // part was itself published. Page boundaries are the natural granularity:
        // a page is the smallest unit the table can hand to another sequence, so
        // publishing there is what lets two conversations share the head they
        // have in common rather than only exact repeats.
        let page_size = paged.cache.page_table.page_size;
        let mut lengths = (1..)
            .map(|pages| pages * page_size)
            .take_while(|&len| len < tokens.len())
            .collect::<Vec<_>>();
        lengths.push(tokens.len());
        for len in lengths {
            if len == 0 {
                continue;
            }
            let key = prefix_key(inputs, &tokens[..len]);
            if paged.prefix.lookup(&key).0 == key.len() {
                continue;
            }
            let pages = sequence_pages_for_len(&paged.cache, seq, len)?;
            paged
                .prefix
                .insert_pages(&key, &pages, &mut paged.cache.page_table);
        }
        if paged.active == Some(seq) {
            paged.active = None;
        }
        for page_id in paged.cache.page_table.remove_sequence(seq) {
            paged.cache.page_table.free(page_id);
        }
        Ok(())
    }

    /// Whether position ids are a plain function of the absolute past length,
    /// and so can be rebuilt after the KV is truncated.
    fn positions_are_linear(&self) -> bool {
        self.models
            .directory
            .spec
            .positions
            .as_ref()
            .is_none_or(|program| {
                program
                    .continuation
                    .as_deref()
                    .is_none_or(|continuation| continuation == "linear_increment")
            })
    }

    /// Whether the decoder's position ids are supplied by a dataflow edge
    /// rather than derived from the absolute past length.
    fn decoder_positions_are_routed(&self, decoder_in_edges: &[(String, String)]) -> bool {
        let Some(position_input) = self
            .decoder_state
            .as_ref()
            .and_then(|state| state.io.position_ids_input.as_deref())
        else {
            return false;
        };
        decoder_in_edges
            .iter()
            .any(|(_, port)| port == position_input)
    }

    /// How many leading prompt tokens the retained decoder KV can serve.
    ///
    /// Zero whenever anything about the request's identity changed, whenever
    /// the prompt shares no leading token with the retained context, or
    /// whenever the attachments could not be digested.
    ///
    /// When the prompt diverges part-way, the retained KV is first truncated to
    /// the shared head. Truncation can decline — an opaque past with no
    /// identifiable sequence axis, or fixed loop-carried state — in which case
    /// nothing is reused and the turn is recomputed.
    fn reusable_prefix_len(&mut self, inputs: Option<Digest>, prompt_tokens: &[TokenId]) -> usize {
        let (Some(inputs), Some(retained)) = (inputs, self.retained.as_ref()) else {
            return 0;
        };
        let retained_len = retained.tokens.len();
        let shared = retained.reusable_prefix(inputs, prompt_tokens);
        let Some(state) = self.decoder_state.as_mut() else {
            return 0;
        };
        if !state.use_kv || shared == 0 {
            return 0;
        }
        if shared == retained_len {
            return shared;
        }
        // Extending keeps the carried position state valid; truncating does not,
        // and only a linear continuation can be rebuilt from the absolute past
        // length alone. A model that carries or is handed its coordinates would
        // resume from positions describing tokens that no longer exist.
        if !self.positions_are_linear() {
            self.decoder_state = None;
            return 0;
        }
        let Some(state) = self.decoder_state.as_mut() else {
            return 0;
        };
        match state.truncate_past(retained_len, shared) {
            Ok(true) => shared,
            // Declining to truncate is a normal outcome, and so is a failed
            // slice: either way the KV is no longer trustworthy for reuse, so
            // the state is dropped and the turn recomputes from scratch.
            _ => {
                self.decoder_state = None;
                0
            }
        }
    }

    /// Post-decode counterpart to [`synthesize`](Self::synthesize) for a
    /// nested-AR (multi-decoder TTS) pipeline: drive the outer/inner loops (which
    /// publish `{outer}.output_codes` into the pool), then run the `final_only`
    /// vocoder stage over the pool to produce the waveform.
    fn synthesize_nested(
        &mut self,
        pipeline_request: PipelineGenerateRequest,
    ) -> anyhow::Result<PipelineSynthesis> {
        let present = pipeline_request.present.clone();
        let post_decode_components = match &self.plan {
            PipelinePlan::NestedAutoregressive(plan) => plan.post_decode_components.clone(),
            _ => anyhow::bail!("internal error: synthesize_nested on a non-nested plan"),
        };
        let (generation, mut tensors) = self.run_nested_autoregressive(pipeline_request)?;
        self.run_prompt_phase_components(
            &post_decode_components,
            &mut tensors,
            "postlogue",
            &present,
            None,
        )?;
        Ok(PipelineSynthesis {
            generation,
            tensors,
        })
    }

    /// Drive a **dual, hierarchically-nested autoregressive** pipeline for the
    /// multi-decoder TTS shape in DESIGN.md §20.3.
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
        let present = pipeline_request.present.clone();
        self.ensure_component_present(&plan.outer, &present, "nested outer decoder")?;
        self.ensure_component_present(&plan.inner, &present, "nested inner decoder")?;

        let options = pipeline_request.request.options.clone();
        options.validate()?;
        let prompt_tokens = tokenize_with(self.tokenizer()?, &pipeline_request.request.prompt)?;
        if prompt_tokens.is_empty() {
            anyhow::bail!("prompt must contain at least one token");
        }

        let mut tensors = self.prepare_request_tensors(pipeline_request.inputs, &present)?;
        self.seed_prompt_token_inputs(&plan.prompt_components, &prompt_tokens, &mut tensors)?;
        // Explicitly seed the prefill embedder's metadata-declared prompt input
        // with the tokenized prompt (int64 `[1, L]`) unless a dataflow edge
        // already routes it. This does NOT rely on `is_token_input_name` — the
        // prompt port is declared in the PrefillEmbedderSpec, never guessed.
        if let Some(prefill) = plan
            .prefill_embedder
            .as_ref()
            .filter(|binding| self.plan.component_is_present(&binding.component, &present))
        {
            let endpoint = format!("{}.{}", prefill.component, prefill.prompt_input);
            let routed = plan.dataflow.iter().any(|edge| edge.to == endpoint);
            if !routed && !tensors.contains_key(&endpoint) {
                let ids: Vec<i64> = prompt_tokens.iter().map(|&t| i64::from(t)).collect();
                let value = Value::from_slice_i64(&ids, &[1, ids.len() as i64])?;
                tensors.insert(endpoint, value);
            }
        }
        self.run_prompt_phase_components(
            &plan.prompt_components,
            &mut tensors,
            "prologue",
            &present,
            None,
        )?;

        // Fixed routed extras for each decoder (encoder conditioning etc.). The
        // inner decoder's seed input is threaded per inner step, so exclude it.
        // In pre-embedder mode the outer decoder's per-step `inputs_embeds` is
        // built each frame (not a fixed routed extra), so exclude it too.
        let outer_extra_exclude = plan.pre_embedder.as_ref().map(|p| p.outer_input.as_str());
        let outer_extras =
            self.decoder_extra_inputs(&plan.outer, &tensors, outer_extra_exclude, &present)?;
        let inner_extras = self.decoder_extra_inputs(
            &plan.inner,
            &tensors,
            Some(&plan.inner_embeds_input),
            &present,
        )?;

        let outer_session = self
            .models
            .session(&plan.outer)
            .with_context(|| format!("nested outer decoder '{}' was not loaded", plan.outer))?;
        let inner_session = self
            .models
            .session(&plan.inner)
            .with_context(|| format!("nested inner decoder '{}' was not loaded", plan.inner))?;

        // Resolve the pre-embedder session and confirm its metadata-declared
        // ports exist (sessions are not available at plan-build time). All port
        // names come from the `PreEmbedderSpec` / the required dataflow edge —
        // there is NO name/dtype guessing here.
        let pre_embed = match plan
            .pre_embedder
            .as_ref()
            .filter(|binding| self.plan.component_is_present(&binding.component, &present))
        {
            Some(binding) => {
                let session = self.models.session(&binding.component).with_context(|| {
                    format!("nested pre_embedder '{}' was not loaded", binding.component)
                })?;
                let frame_codes_input = binding.frame_codes_input.clone();
                if !session
                    .inputs()
                    .iter()
                    .any(|info| info.name == frame_codes_input)
                {
                    anyhow::bail!(
                        "nested pre_embedder '{}' has no declared frame_codes input '{}'",
                        binding.component,
                        frame_codes_input
                    );
                }
                let text_embed_input = binding.text_embed_input.clone();
                if let Some(name) = &text_embed_input
                    && !session.inputs().iter().any(|info| &info.name == name)
                {
                    anyhow::bail!(
                        "nested pre_embedder '{}' has no declared text_embed input '{}'",
                        binding.component,
                        name
                    );
                }
                if !session
                    .output_names()
                    .iter()
                    .any(|name| name == &binding.output_port)
                {
                    anyhow::bail!(
                        "nested pre_embedder '{}' has no declared output port '{}'",
                        binding.component,
                        binding.output_port
                    );
                }
                // Hidden size for the per-step embedding / zero `text_embed`:
                // prefer the outer decoder's `inputs_embeds` input (a metadata
                // port captured from the edge `to`), fall back to the
                // pre-embedder's declared output port.
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
                            .find(|info| info.name == binding.output_port)
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
                    output_port: binding.output_port.clone(),
                    frame_codes_input,
                    text_embed_input,
                    hidden,
                })
            }
            None => None,
        };

        // Resolve the optional prefill embedder's pooled outputs (it ran as a
        // prompt-phase component above, seeded with the tokenized prompt via
        // `seed_prompt_token_inputs`). `prefill_embeds` [1, prefill_len, hidden]
        // seeds the talker's frame-0 `inputs_embeds` DIRECTLY (multi-position
        // PREFILL); `trailing_text_embeds` [1, trailing_len, hidden] supplies one
        // `text_embed` vector per outer frame `k >= 1` (fed through the
        // pre-embedder). Only valid alongside `pre_embedder`.
        let prefill = match plan
            .prefill_embedder
            .as_ref()
            .filter(|binding| self.plan.component_is_present(&binding.component, &present))
        {
            Some(binding) => {
                let component = binding.component.as_str();
                let pre = pre_embed.as_ref().with_context(|| {
                    format!(
                        "nested prefill_embedder '{component}' requires a pre_embedder to be set"
                    )
                })?;
                let _ = self.models.session(component).with_context(|| {
                    format!("nested prefill_embedder '{component}' was not loaded")
                })?;
                // The prefill component's two float outputs are metadata-declared
                // (`prefill_output` / `trailing_output`) — no name/dtype guessing.
                let prefill_name = binding.prefill_output.as_str();
                let trailing_name = binding.trailing_output.as_str();
                let prefill_value = tensors
                    .get(&format!("{component}.{prefill_name}"))
                    .with_context(|| {
                        format!(
                            "nested prefill_embedder '{component}' produced no pooled \
                             '{prefill_name}' output (did it run in the prompt phase?)"
                        )
                    })?;
                let prefill_len = match prefill_value.shape() {
                    [1, p, _] if *p > 0 => *p as usize,
                    other => anyhow::bail!(
                        "nested prefill_embedder '{component}' '{prefill_name}' must be \
                         [1, prefill_len, hidden]; got {other:?}"
                    ),
                };
                let prefill_embeds = clone_value(prefill_value)?;
                let trailing_value = tensors
                    .get(&format!("{component}.{trailing_name}"))
                    .with_context(|| {
                        format!(
                            "nested prefill_embedder '{component}' produced no pooled \
                             '{trailing_name}' output (did it run in the prompt phase?)"
                        )
                    })?;
                let trailing_len = match trailing_value.shape() {
                    [1, t, h] if *h as usize == pre.hidden => *t as usize,
                    other => anyhow::bail!(
                        "nested prefill_embedder '{component}' '{trailing_name}' must be \
                         [1, trailing_len, {}]; got {other:?}",
                        pre.hidden
                    ),
                };
                let trailing = trailing_value.to_vec_f32_lossy().map_err(|e| {
                    anyhow::anyhow!("failed to read trailing_text_embeds tensor: {e}")
                })?;
                Some(ResolvedPrefill {
                    prefill_embeds,
                    prefill_len,
                    trailing,
                    trailing_len,
                    hidden: pre.hidden,
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
                // Build (or, on frame 0 with a prefill embedder, look up) the
                // talker's per-step `inputs_embeds`.
                let (inputs_embeds, positions) =
                    if let Some(prefill) = prefill.as_ref().filter(|_| _frame == 0) {
                        // Frame 0 PREFILL: feed the prefill embedder's multi-position
                        // `prefill_embeds` DIRECTLY to the talker (do NOT run the
                        // pre-embedder), advancing the KV past by `prefill_len`.
                        (clone_value(&prefill.prefill_embeds)?, prefill.prefill_len)
                    } else {
                        // Build this frame's `frame_codes` from the previous frame's
                        // code tuple (frame 0 without a prefill embedder uses a zero
                        // seed), run the pre-embedder to materialize a single-position
                        // `inputs_embeds`. With a prefill embedder, frames `k >= 1`
                        // feed `text_embed = trailing_text_embeds[:, k-1, :]` (zeros
                        // once the trailing text is exhausted — a close stand-in for
                        // the reference's tts_pad embedding; exact tts_pad is a
                        // documented refinement).
                        let frame_codes = prev_frame_codes
                            .clone()
                            .unwrap_or_else(|| vec![0i64; plan.num_code_groups]);
                        let text_embed = match prefill.as_ref() {
                            Some(prefill) => {
                                let idx = _frame - 1;
                                let hidden = prefill.hidden;
                                let slice = if idx < prefill.trailing_len {
                                    prefill.trailing[idx * hidden..(idx + 1) * hidden].to_vec()
                                } else {
                                    vec![0.0f32; hidden]
                                };
                                Some(slice)
                            }
                            None => None,
                        };
                        (
                            run_pre_embedder(pre, &frame_codes, text_embed.as_deref())?,
                            1,
                        )
                    };
                let mut step_extras = Vec::with_capacity(outer_extras.len() + 1);
                for (name, value) in &outer_extras {
                    step_extras.push((name.clone(), clone_value(value)?));
                }
                step_extras.push((pre.outer_input.clone(), inputs_embeds));
                // Match the token-position count to the fed `inputs_embeds`
                // sequence length so any position_ids/attention_mask the talker
                // exposes stay consistent (the talker itself is embeds-driven and
                // ignores the token ids).
                let position_tokens = vec![0u32; positions];
                let outputs = run_decode_step_with_extra(
                    outer_session,
                    &mut outer_state,
                    &position_tokens,
                    outer_past_len,
                    &step_extras,
                )?;
                outer_past_len += positions;
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
            // inner residuals for groups 1..G-1, where code_0 comes from the
            // talker rather than the code predictor.
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
    fn run_iterative(&self, request: PipelineGenerateRequest) -> anyhow::Result<PipelineTensors> {
        let PipelinePlan::Iterative(plan) = &self.plan else {
            anyhow::bail!("internal error: run_iterative on a non-iterative plan");
        };

        // Live overrides (ComfyUI-style): re-drive the already-loaded models with
        // different loop parameters, no reload. Seed / prompt / negative are
        // already live via per-request inputs, so only loop params are overridden.
        let present = request.present.clone();
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
        let mut constants = self.prepare_request_tensors(request.inputs, &present)?;
        let mut stage_timings: Vec<serde_json::Value> = Vec::new();
        {
            let _span = onnx_genai_ort::prof_span!("diffusion.text_encode");
            self.run_prompt_phase_components(
                &plan.prompt_components,
                &mut constants,
                "encode",
                &present,
                Some(&mut stage_timings),
            )?;
        }
        if !self.plan.component_is_present(&plan.denoiser, &present) {
            {
                let _span = onnx_genai_ort::prof_span!("diffusion.vae_decode");
                self.run_prompt_phase_components(
                    &plan.final_components,
                    &mut constants,
                    "decode",
                    &present,
                    Some(&mut stage_timings),
                )?;
            }
            dump_stage_timings(&stage_timings);
            return Ok(constants);
        }

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
        {
            let _denoise_loop_span = onnx_genai_ort::prof_span!("diffusion.denoise_loop");
            for step in start_step..num_steps {
                let _step_span =
                    onnx_genai_ort::prof_span!("diffusion.denoise_step", "step" => step);
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
                            format!(
                                "loop-carried input '{}.{in_port}' was not produced",
                                plan.denoiser
                            )
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
                            format!(
                                "unconditional pass did not produce '{}.{port}'",
                                plan.denoiser
                            )
                        })?;
                        let cond_v = cond_value.to_vec_f32_lossy()?;
                        let uncond_v = uncond_value.to_vec_f32_lossy()?;
                        let guided: Vec<f32> = uncond_v
                            .iter()
                            .zip(&cond_v)
                            .map(|(u, c)| u + scale * (c - u))
                            .collect();
                        combined.insert(
                            port.clone(),
                            Value::from_slice_f32(&guided, cond_value.shape())?,
                        );
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
                        format!(
                            "denoiser did not produce loop output '{}.{out_port}'",
                            plan.denoiser
                        )
                    })?;
                    let next = if let Some(scheduler) = scheduler {
                        let _scheduler_span =
                            onnx_genai_ort::prof_span!("diffusion.scheduler_step", "step" => step);
                        let sample = raw_samples.get(in_port).with_context(|| {
                            format!(
                                "missing loop-carried sample for '{}.{in_port}'",
                                plan.denoiser
                            )
                        })?;
                        if scheduler.needs_noise() {
                            let noise = self
                                .step_noise(plan, num_steps, &constants, in_port, step, sample)?;
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
        {
            let _span = onnx_genai_ort::prof_span!("diffusion.vae_decode");
            self.run_prompt_phase_components(
                &plan.final_components,
                &mut tensors,
                "decode",
                &present,
                Some(&mut stage_timings),
            )?;
        }
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
                inputs.push((
                    port.to_string(),
                    coerce_value_to_dtype(over_value, info.dtype)?,
                ));
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
                    carried.get(port).with_context(|| {
                        format!("loop-carried input '{endpoint}' was not produced")
                    })?
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
        let _span = onnx_genai_ort::prof_span!("diffusion.denoiser_pass", "step" => step);
        let outputs = denoiser.run(&refs).map_err(|e| {
            anyhow::anyhow!(
                "ORT denoiser '{}' failed at step {step}: {e}",
                plan.denoiser
            )
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
    fn run_composite(&self, request: PipelineGenerateRequest) -> anyhow::Result<PipelineTensors> {
        let PipelinePlan::Composite(plan) = &self.plan else {
            anyhow::bail!("internal error: run_composite on a non-composite plan");
        };
        let present = request.present;
        let mut tensors = self.prepare_request_tensors(request.inputs, &present)?;
        for stage in &plan.stages {
            match &stage.kind {
                CompositeStageKind::SinglePass { model } => {
                    self.run_prompt_phase_components(
                        std::slice::from_ref(model),
                        &mut tensors,
                        &stage.name,
                        &present,
                        None,
                    )?;
                }
            }
        }
        Ok(tensors)
    }

    fn run_single_pass(&self, request: PipelineGenerateRequest) -> anyhow::Result<PipelineTensors> {
        let PipelinePlan::SinglePass(plan) = &self.plan else {
            anyhow::bail!("internal error: run_single_pass on a non-single-pass plan");
        };
        let present = request.present;
        let mut tensors = self.prepare_request_tensors(request.inputs, &present)?;
        self.run_prompt_phase_components(
            &plan.prompt_components,
            &mut tensors,
            "prologue",
            &present,
            None,
        )?;

        if !self.plan.component_is_present(&plan.model, &present) {
            return Ok(tensors);
        }

        let model = self
            .models
            .session(&plan.model)
            .with_context(|| format!("pipeline model '{}' was not loaded", plan.model))?;
        let inputs = self.component_inputs(&plan.model, model, &tensors, &present)?;
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

    fn prepare_request_tensors(
        &self,
        inputs: PipelineTensors,
        present: &BTreeSet<String>,
    ) -> anyhow::Result<PipelineTensors> {
        if present.iter().any(String::is_empty) {
            anyhow::bail!("pipeline request presence keys must be non-empty");
        }
        let mut dimensions = HashMap::<String, i64>::new();

        for (component, model) in &self.models.directory.spec.models {
            let Some(io) = model.io.as_ref() else {
                continue;
            };
            let session = self
                .models
                .session(component)
                .with_context(|| format!("pipeline component '{component}' was not loaded"))?;
            for (port, optional) in &io.optional_inputs {
                let endpoint = format!("{component}.{port}");
                let route = self.plan.dataflow().iter().find(|edge| edge.to == endpoint);
                let supplied_endpoint = inputs
                    .get(&endpoint)
                    .map(|value| (endpoint.as_str(), value));
                let supplied_route = route.and_then(|edge| {
                    inputs
                        .get(&edge.from)
                        .map(|value| (edge.from.as_str(), value))
                });
                let supplied = supplied_endpoint.or(supplied_route);
                let is_present = present.contains(&optional.presence);

                if !is_present {
                    if let Some((supplied_name, _)) = supplied {
                        anyhow::bail!(
                            "pipeline input '{supplied_name}' is associated with presence key '{}' \
                             but that key was declared absent",
                            optional.presence
                        );
                    }
                } else if supplied.is_none() {
                    let active_route = route.is_some_and(|edge| {
                        endpoint_component(&edge.from).is_some_and(|producer| {
                            self.plan.component_is_present(producer, present)
                        })
                    });
                    if !active_route {
                        anyhow::bail!(
                            "missing optional-but-present pipeline input '{endpoint}' for presence \
                             key '{}': supply the destination endpoint or an active routed source",
                            optional.presence
                        );
                    }
                }

                let info = session
                    .inputs()
                    .iter()
                    .find(|info| info.name == *port)
                    .with_context(|| {
                        format!(
                            "optional pipeline input '{endpoint}' is not exposed by its ONNX graph"
                        )
                    })?;
                if info.shape.len() != optional.absent.shape.len() {
                    anyhow::bail!(
                        "invalid fallback for optional pipeline input '{endpoint}': declared rank {} \
                         does not match graph rank {}",
                        optional.absent.shape.len(),
                        info.shape.len()
                    );
                }
                for (index, dimension) in optional.absent.shape.iter().enumerate() {
                    let TensorDimension::Symbol(symbol) = dimension else {
                        continue;
                    };
                    if info.shape[index] >= 0 {
                        bind_dimension(&mut dimensions, symbol, info.shape[index], &endpoint)?;
                    }
                    if let Some((_, value)) = supplied {
                        if value.shape().len() != optional.absent.shape.len() {
                            anyhow::bail!(
                                "pipeline input '{endpoint}' has rank {}, expected {} from its \
                                 optional-input contract",
                                value.shape().len(),
                                optional.absent.shape.len()
                            );
                        }
                        bind_dimension(&mut dimensions, symbol, value.shape()[index], &endpoint)?;
                    }
                }
            }
        }

        let mut tensors = inputs;
        for (component, model) in &self.models.directory.spec.models {
            let Some(io) = model.io.as_ref() else {
                continue;
            };
            let session = self
                .models
                .session(component)
                .with_context(|| format!("pipeline component '{component}' was not loaded"))?;
            for (port, optional) in &io.optional_inputs {
                if present.contains(&optional.presence) {
                    continue;
                }
                let endpoint = format!("{component}.{port}");
                if tensors.contains_key(&endpoint) {
                    continue;
                }
                let info = session
                    .inputs()
                    .iter()
                    .find(|info| info.name == *port)
                    .with_context(|| {
                        format!(
                            "optional pipeline input '{endpoint}' is not exposed by its ONNX graph"
                        )
                    })?;
                let shape = optional
                    .absent
                    .shape
                    .iter()
                    .map(|dimension| match dimension {
                        TensorDimension::Fixed(value) => Ok(*value),
                        TensorDimension::Symbol(symbol) => {
                            dimensions.get(symbol).copied().with_context(|| {
                                format!(
                                    "unresolved fallback shape symbol '{symbol}' for optional \
                                     pipeline input '{endpoint}'"
                                )
                            })
                        }
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?;
                let value = match optional.absent.kind {
                    AbsentInputKind::Zeros => zero_value(&shape, info.dtype).with_context(|| {
                        format!(
                            "invalid fallback for optional pipeline input '{endpoint}' with dtype \
                             {:?} and shape {shape:?}",
                            info.dtype
                        )
                    })?,
                };
                tensors.insert(endpoint, value);
            }
        }
        Ok(tensors)
    }

    fn ensure_component_present(
        &self,
        component: &str,
        present: &BTreeSet<String>,
        role: &str,
    ) -> anyhow::Result<()> {
        if let Some(key) = self.plan.presence_condition(component)
            && !present.contains(key)
        {
            anyhow::bail!(
                "{role} '{component}' is gated by absent presence key '{key}' and cannot execute"
            );
        }
        Ok(())
    }

    fn missing_input_error(
        &self,
        component: &str,
        port: &str,
        present: &BTreeSet<String>,
    ) -> anyhow::Error {
        let endpoint = format!("{component}.{port}");
        let optional = self
            .models
            .directory
            .spec
            .models
            .get(component)
            .and_then(|model| model.io.as_ref())
            .and_then(|io| io.optional_inputs.get(port));
        match optional {
            Some(optional) if present.contains(&optional.presence) => anyhow::anyhow!(
                "missing optional-but-present pipeline input '{endpoint}' for presence key '{}'",
                optional.presence
            ),
            Some(optional) => anyhow::anyhow!(
                "missing or invalid fallback for absent optional pipeline input '{endpoint}' \
                 (presence key '{}')",
                optional.presence
            ),
            None => anyhow::anyhow!("missing required pipeline input '{endpoint}'"),
        }
    }

    fn run_prompt_phase_components(
        &self,
        components: &[String],
        tensors: &mut PipelineTensors,
        phase: &str,
        present: &BTreeSet<String>,
        mut timings: Option<&mut Vec<serde_json::Value>>,
    ) -> anyhow::Result<()> {
        for component in components {
            if !self.plan.component_is_present(component, present) {
                continue;
            }
            let session = self
                .models
                .session(component)
                .with_context(|| format!("pipeline component '{component}' was not loaded"))?;
            let inputs = self.component_inputs(component, session, tensors, present)?;

            // A prompt-phase component is a pure function of its inputs — that
            // is what separates it from an `every_step` component — so identical
            // input bytes mean identical outputs. Re-asking about the same image
            // then costs a hash instead of a vision encoder forward pass.
            let memoizable = self.component_cache.borrow().is_enabled()
                && self.memoizable_components.contains(component);
            let key = memoizable.then(|| {
                digest_named_values(
                    component,
                    inputs.iter().map(|(name, value)| (name.as_str(), value)),
                )
            });
            let key = match key {
                Some(Some(key)) => Some(key),
                // Enabled but undigestible: run without touching the cache
                // rather than key on a partial description of the inputs.
                Some(None) => {
                    self.component_cache.borrow_mut().note_unkeyable();
                    None
                }
                None => None,
            };
            if let Some(key) = key
                && let Some(cached) = self.component_cache.borrow_mut().get(key)
            {
                if let Some(sink) = timings.as_deref_mut() {
                    sink.push(serde_json::json!({
                        "component": component,
                        "phase": phase,
                        "ms": 0.0,
                        "cached": true,
                    }));
                }
                for (name, value) in cached {
                    tensors.insert(format!("{component}.{name}"), value);
                }
                continue;
            }

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
            let named = session
                .output_names()
                .iter()
                .cloned()
                .zip(outputs)
                .collect::<Vec<_>>();
            if let Some(key) = key {
                let mut cache = self.component_cache.borrow_mut();
                cache.note_miss();
                cache.insert(key, &named);
            }
            for (name, value) in named {
                tensors.insert(format!("{component}.{name}"), value);
            }
        }
        Ok(())
    }

    /// KV page counters, when the decoder's KV is paged.
    ///
    /// `None` for a decoder whose KV cannot be paged, rather than zeros, which
    /// would read as "a page pool that did nothing".
    pub fn page_stats(&self) -> Option<onnx_genai_kv::PageStats> {
        self.paged
            .as_ref()
            .map(|paged| paged.cache.page_table.stats())
    }

    /// What the KV page pool is holding right now, when the decoder pages.
    pub fn page_usage(&self) -> Option<onnx_genai_kv::PageUsage> {
        self.paged
            .as_ref()
            .map(|paged| paged.cache.page_table.usage())
    }

    /// Counters describing what the pipeline's reuse caches did.
    pub fn cache_stats(&self) -> PipelineCacheStats {
        self.component_cache.borrow().stats()
    }

    /// Clear the per-generation counters reported by [`cache_stats`](Self::cache_stats).
    pub fn reset_cache_stats(&self) {
        self.component_cache.borrow_mut().reset_stats();
    }

    /// Digest everything about a request that changes what the decoder computes.
    ///
    /// This is the part of a multimodal prompt's identity that token ids cannot
    /// express: placeholder expansion turns any image into the same repeated
    /// token, so two different pictures produce byte-identical prompts. Without
    /// this digest in the key, retained KV for one photo would be served for
    /// another and the model would answer confidently about a picture it never
    /// saw.
    ///
    /// Covers the bound tensors, the presence keys, and the tile count.
    ///
    /// `None` when some input cannot be digested, which disables reuse for the
    /// request rather than keying it on an incomplete description.
    fn digest_request_identity(request: &PipelineGenerateRequest) -> Option<Digest> {
        let mut builder = DigestBuilder::new();

        // Presence keys gate which components run and which optional decoder
        // inputs are bound, so the same tensors under different presence keys
        // are a different computation and must not share KV.
        builder.absorb_u64(request.present.len() as u64);
        for key in &request.present {
            builder.absorb_str(key);
        }
        // Tile count drives placeholder expansion for encoder-free multimodal
        // pipelines, and so the meaning of the prompt's placeholder run.
        builder.absorb_u64(request.num_image_tiles.unwrap_or(0) as u64);

        let mut endpoints = request.inputs.keys().collect::<Vec<_>>();
        endpoints.sort();
        builder.absorb_u64(endpoints.len() as u64);
        for endpoint in endpoints {
            builder.absorb_str(endpoint);
            if !absorb_value(&mut builder, &request.inputs[endpoint]) {
                return None;
            }
        }
        Some(builder.finish())
    }

    fn component_inputs(
        &self,
        component: &str,
        session: &Session,
        tensors: &PipelineTensors,
        present: &BTreeSet<String>,
    ) -> anyhow::Result<Vec<(String, Value)>> {
        let mut inputs = Vec::new();
        for info in session.inputs() {
            let endpoint = format!("{component}.{}", info.name);
            let routed = self
                .plan
                .dataflow()
                .iter()
                .find(|edge| {
                    edge.to == endpoint
                        && endpoint_component(&edge.from)
                            .is_none_or(|source| self.plan.component_is_present(source, present))
                })
                .and_then(|edge| tensors.get(&edge.from));
            let value = tensors
                .get(&endpoint)
                .or(routed)
                .ok_or_else(|| self.missing_input_error(component, &info.name, present))?;
            inputs.push((info.name.clone(), coerce_value_to_dtype(value, info.dtype)?));
        }
        Ok(inputs)
    }

    fn decoder_extra_inputs(
        &self,
        decoder: &str,
        tensors: &PipelineTensors,
        exclude_input: Option<&str>,
        present: &BTreeSet<String>,
    ) -> anyhow::Result<Vec<(String, Value)>> {
        let mut extras = Vec::new();
        let mut bound = BTreeSet::new();
        for edge in self
            .plan
            .edges_to_component(decoder)
            .filter(|edge| endpoint_component(&edge.from).is_some_and(|from| from != decoder))
            .filter(|edge| {
                endpoint_component(&edge.from)
                    .is_none_or(|source| self.plan.component_is_present(source, present))
            })
        {
            let (_, input) = parse_endpoint(&edge.to)?;
            // The per-step `inputs_embeds` edge is threaded dynamically by the
            // decode loop (re-embedding each step), not carried as a fixed extra.
            if exclude_input == Some(input) {
                continue;
            }
            let value = tensors
                .get(&edge.to)
                .or_else(|| tensors.get(&edge.from))
                .with_context(|| {
                    format!(
                        "missing pipeline tensor '{}' and routed source '{}'",
                        edge.to, edge.from
                    )
                })?;
            extras.push((input.to_string(), clone_value(value)?));
            bound.insert(input.to_string());
        }
        if let Some(optional_inputs) = self
            .models
            .directory
            .spec
            .models
            .get(decoder)
            .and_then(|model| model.io.as_ref())
            .map(|io| &io.optional_inputs)
        {
            let session = self
                .models
                .session(decoder)
                .with_context(|| format!("pipeline decoder '{decoder}' was not loaded"))?;
            for port in optional_inputs.keys() {
                if exclude_input == Some(port.as_str()) || bound.contains(port) {
                    continue;
                }
                let endpoint = format!("{decoder}.{port}");
                let value = tensors
                    .get(&endpoint)
                    .ok_or_else(|| self.missing_input_error(decoder, port, present))?;
                let dtype = session
                    .inputs()
                    .iter()
                    .find(|info| info.name == *port)
                    .with_context(|| {
                        format!("optional pipeline input '{endpoint}' is not exposed by its graph")
                    })?
                    .dtype;
                extras.push((port.clone(), coerce_value_to_dtype(value, dtype)?));
            }
        }
        Ok(extras)
    }

    /// Seed the prompt token ids into the shared pool for any prompt-phase
    /// component that consumes a token input (`input_ids`) which is neither
    /// supplied by the caller nor routed by a dataflow edge.
    ///
    /// The token port is taken only from explicit component `io.token_input`
    /// metadata. Components without that declaration are not implicitly seeded.
    fn seed_prompt_token_inputs(
        &self,
        components: &[String],
        prompt_tokens: &[TokenId],
        tensors: &mut PipelineTensors,
    ) -> anyhow::Result<()> {
        for component in components {
            let Some(token_input) = self
                .models
                .directory
                .spec
                .models
                .get(component)
                .and_then(|model| model.io.as_ref())
                .and_then(|io| io.token_input.as_deref())
            else {
                continue;
            };
            let endpoint = format!("{component}.{token_input}");
            let routed = self.plan.dataflow().iter().any(|edge| edge.to == endpoint);
            if routed || tensors.contains_key(&endpoint) {
                continue;
            }
            let ids: Vec<i64> = prompt_tokens.iter().map(|&t| i64::from(t)).collect();
            let value = Value::from_slice_i64(&ids, &[1, ids.len() as i64])?;
            tensors.insert(endpoint, value);
        }
        Ok(())
    }

    /// Resolve the STATIC encoder-produced cross-attention KV tensors that feed
    /// the autoregressive decoder on every step.
    ///
    /// For an encoder-decoder (e.g. Whisper) pipeline the encoder runs once as a
    /// prompt-phase prologue and publishes its `present_*_cross_%d` outputs into
    /// the shared pool as `{encoder}.present_*_cross_%d`. Those tensors encode
    /// the whole audio/text prompt and are STATIC for the entire decode: they
    /// never grow or change across autoregressive steps (unlike the decoder's
    /// self-attention KV cache). They are therefore cloned once here and re-bound
    /// verbatim to the decoder's `past_*_cross_%d` inputs on every step, rather
    /// than recomputed. The pairing comes from the decoder's declared
    /// `cross_kv_inputs`/`cross_kv_outputs` (resolved into `cross_kv_pairs`),
    /// keyed off the encoder-decoder pipeline shape, not any model name.
    fn static_cross_kv_bindings(
        &self,
        cross_kv_pairs: &[(String, String)],
        tensors: &PipelineTensors,
    ) -> anyhow::Result<Vec<(String, Arc<Value>)>> {
        let mut bindings = Vec::with_capacity(cross_kv_pairs.len());
        for (decoder_input, encoder_output) in cross_kv_pairs {
            let suffix = format!(".{encoder_output}");
            let mut matches = tensors
                .iter()
                .filter(|(key, _)| key.ends_with(&suffix) || key.as_str() == encoder_output);
            let (_, value) = matches.next().with_context(|| {
                format!(
                    "encoder-decoder cross-attention: no pooled encoder output '{encoder_output}' \
                     to bind decoder input '{decoder_input}'; the encoder prologue must run and \
                     publish it before decode"
                )
            })?;
            if matches.next().is_some() {
                anyhow::bail!(
                    "encoder-decoder cross-attention: multiple pooled tensors match encoder output \
                     '{encoder_output}' for decoder input '{decoder_input}'; the producing \
                     component is ambiguous"
                );
            }
            // `Arc<Value>` mirrors the shared-ownership convention the ORT
            // decode paths already use for per-step-invariant tensors; `Value`
            // is neither `Send` nor `Sync`, so the lint is suppressed here as it
            // is in `onnx-genai-ort`.
            #[allow(clippy::arc_with_non_send_sync)]
            let shared = Arc::new(clone_value(value)?);
            bindings.push((decoder_input.clone(), shared));
        }
        Ok(bindings)
    }

    /// Precompute the static routing edges feeding the autoregressive `decoder`.
    ///
    /// Returns `(source_endpoint, decoder_input_port)` for every dataflow edge
    /// into the decoder whose source is a **different** component (self-edges are
    /// loop-carried KV / recurrent state, resolved inside the decode step). Both
    /// `every_step` producers and cached `prompt_only` conditioning route through
    /// this list; the values are re-read from the shared pool on every step, so
    /// per-step outputs stay fresh while fixed conditioning is simply reused.
    fn decoder_in_edges(
        &self,
        decoder: &str,
        present: &BTreeSet<String>,
        tensors: &PipelineTensors,
    ) -> anyhow::Result<Vec<(String, String)>> {
        let mut edges = Vec::new();
        let mut bound = BTreeSet::new();
        for edge in self
            .plan
            .edges_to_component(decoder)
            .filter(|edge| endpoint_component(&edge.from).is_some_and(|from| from != decoder))
            .filter(|edge| {
                endpoint_component(&edge.from)
                    .is_none_or(|source| self.plan.component_is_present(source, present))
            })
        {
            let (_, input) = parse_endpoint(&edge.to)?;
            let source = if tensors.contains_key(&edge.to) {
                edge.to.clone()
            } else {
                edge.from.clone()
            };
            edges.push((source, input.to_string()));
            bound.insert(input.to_string());
        }
        if let Some(io) = self
            .models
            .directory
            .spec
            .models
            .get(decoder)
            .and_then(|model| model.io.as_ref())
        {
            for port in io.optional_inputs.keys() {
                if bound.contains(port) {
                    continue;
                }
                let endpoint = format!("{decoder}.{port}");
                if tensors.contains_key(&endpoint) {
                    edges.push((endpoint, port.clone()));
                }
            }
        }
        Ok(edges)
    }

    /// Bind each declared `every_step` component to its generic input contract.
    ///
    /// The single running-token port comes from the component's explicit
    /// `io.token_input` metadata — never a tensor-name heuristic. Every other
    /// input is resolved from the shared pool on each step (directly by endpoint
    /// or through a dataflow edge), so cross-conditioning (e.g. image features)
    /// and chained per-step outputs both work without special-casing. This is
    /// the generic replacement for the former one-output `inputs_embeds` fusion
    /// binding: on prefill every component runs over the full prompt, on decode
    /// over the single running token, and all of its outputs are published back
    /// into the pool for routing into the decoder. Returns owned bindings so the
    /// caller can pair each with its loaded session without extending the borrow.
    fn build_step_bindings(
        &self,
        step_components: &[String],
        present: &BTreeSet<String>,
    ) -> anyhow::Result<Vec<StepComponentBinding>> {
        let mut bindings = Vec::with_capacity(step_components.len());
        for component in step_components {
            if !self.plan.component_is_present(component, present) {
                continue;
            }
            let session = self.models.session(component).with_context(|| {
                format!("pipeline every_step component '{component}' was not loaded")
            })?;
            let token_input = self
                .models
                .directory
                .spec
                .models
                .get(component)
                .and_then(|spec| spec.io.as_ref())
                .and_then(|io| io.token_input.clone());
            if let Some(port) = &token_input
                && !session.inputs().iter().any(|info| &info.name == port)
            {
                anyhow::bail!(
                    "every_step component '{component}' declares io.token_input '{port}' but \
                     the graph does not expose it; graph inputs: {:?}",
                    session.input_names()
                );
            }
            let mut routed_inputs = Vec::new();
            for info in session.inputs() {
                if token_input.as_deref() == Some(info.name.as_str()) {
                    continue;
                }
                let endpoint = format!("{component}.{}", info.name);
                let routed_from = self
                    .plan
                    .dataflow()
                    .iter()
                    .find(|edge| {
                        edge.to == endpoint
                            && endpoint_component(&edge.from).is_none_or(|source| {
                                self.plan.component_is_present(source, present)
                            })
                    })
                    .map(|edge| edge.from.clone());
                routed_inputs.push(StepComponentInput {
                    port: info.name.clone(),
                    endpoint,
                    routed_from,
                    dtype: info.dtype,
                    missing_message: self
                        .missing_input_error(component, &info.name, present)
                        .to_string(),
                });
            }
            bindings.push(StepComponentBinding {
                component: component.clone(),
                token_input,
                routed_inputs,
            });
        }
        Ok(bindings)
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
        anyhow::bail!("pipeline metadata tokens_per_tile is 0; must be at least 1");
    }

    let placeholder_id: TokenId = u32::try_from(placeholder_i64).with_context(|| {
        format!("image_placeholder_token_id {placeholder_i64} is out of range for token ID (u32)")
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

    let expansion: usize = tokens_per_tile.checked_mul(num_tiles).context(
        "image token expansion overflow: tokens_per_tile * num_image_tiles is too large",
    )?;

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

/// One generic `every_step` pipeline component and its declared input contract.
///
/// Built by [`PipelineEngine::build_step_bindings`]. On every autoregressive
/// step the component runs over the current token seed (the full prompt during
/// prefill, the single running token during decode) and all of its outputs are
/// published back into the shared pool, from where the decoder's routed inputs
/// are refreshed. This is the architecture-neutral replacement for the former
/// one-output `inputs_embeds` fusion special case: it refreshes every declared
/// sequence-dependent output, and never inspects tensor names to decide roles.
struct StepComponentBinding {
    /// Component name (pool endpoints are `component.port`).
    component: String,
    /// The port seeded with the running token(s), from explicit `io.token_input`
    /// metadata. `None` when the component takes no running-token input (all of
    /// its inputs are routed / fixed conditioning).
    token_input: Option<String>,
    /// Every non-token input, resolved from the shared pool on each step.
    routed_inputs: Vec<StepComponentInput>,
}

/// A single non-token input of a [`StepComponentBinding`], resolved from the
/// shared pool each step (directly by `endpoint`, else via the `routed_from`
/// dataflow source).
struct StepComponentInput {
    /// Graph input port name on the component.
    port: String,
    /// This port's own pool endpoint (`component.port`).
    endpoint: String,
    /// Dataflow-edge source endpoint feeding this port, if any.
    routed_from: Option<String>,
    /// Declared graph-input dtype to coerce the routed value to.
    dtype: DataType,
    /// Presence-aware diagnostic if neither direct nor routed binding exists.
    missing_message: String,
}

/// Whether this error is the KV page pool being full, rather than a fault.
///
/// A full pool is a capacity condition the caller can degrade around; anything
/// else means the mirror is broken and must not be swallowed.
fn is_kv_out_of_memory(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        matches!(
            cause.downcast_ref::<onnx_genai_kv::KvError>(),
            Some(onnx_genai_kv::KvError::OutOfMemory { .. })
        )
    })
}

impl PipelinePagedKv {
    /// Release the in-flight sequence, if any, returning its pages to the pool.
    ///
    /// Pages the prefix cache published are retained by it, so this frees only
    /// what nothing else refers to.
    fn discard_active(&mut self) {
        let Some(seq) = self.active.take() else {
            return;
        };
        for page_id in self.cache.page_table.remove_sequence(seq) {
            self.cache.page_table.free(page_id);
        }
    }

    /// Return pages to the pool by evicting unreferenced cached prefixes,
    /// least-recently-used first.
    ///
    /// Without this the cache would hold every prefix it ever published and the
    /// pool would run dry after enough distinct conversations. Only prefixes no
    /// live sequence is borrowing can go.
    fn evict_until_free(&mut self, wanted_pages: usize) {
        let free = self
            .cache
            .page_table
            .free_count(onnx_genai_kv::Device::Gpu(0));
        if free >= wanted_pages {
            return;
        }
        self.prefix
            .evict_lru(wanted_pages - free, &mut self.cache.page_table);
    }
}

/// Where a decode step's KV is written so later requests can share it.
struct PagedMirror<'a> {
    kv_model: &'a KvModelInfo,
    cache: &'a mut PagedKvCache,
    seq: SequenceId,
    /// Tokens whose KV actually reached the pages so far.
    ///
    /// Mirroring can stop early when the pool runs dry, and only this many
    /// tokens may then be published — the pages beyond it do not exist, and a
    /// key claiming them would hand a later request KV that was never written.
    mirrored_tokens: usize,
    /// Set once the pool refused a page, after which mirroring stops for the
    /// rest of this generation.
    exhausted: bool,
    /// Set once the sliding window dropped pages from this sequence.
    ///
    /// What remains is then `[sinks | recent window]`, which is not a prefix of
    /// anything. Publishing it under a key that says "the first N tokens" would
    /// hand a later request pages with a hole in the middle, so nothing from a
    /// sequence that has been windowed may be published at all.
    windowed: bool,
}

struct PipelineDecodeLoopBackend<'a> {
    decoder: &'a Session,
    decoder_state: &'a mut DecodeState,
    /// Shared tensor pool: external inputs + prompt-phase outputs + the
    /// per-step outputs of the `every_step` components (refreshed each step).
    pool: &'a mut PipelineTensors,
    /// Declared `every_step` components (with their loaded sessions), executed in
    /// topological order on every step before the decoder runs.
    step_components: Vec<(StepComponentBinding, &'a Session)>,
    /// `(source_endpoint, decoder_input_port)` routing recomputed each step.
    decoder_in_edges: Vec<(String, String)>,
    /// Static encoder-produced cross-attention KV bound to the decoder every
    /// step: `(decoder_input_port, shared_value)`. Resolved once from the encoder
    /// prologue outputs and held behind an `Arc` so each step re-binds it as a
    /// no-copy alias (O(1)) rather than deep-copying the large invariant buffer
    /// (see `PipelineEngine::static_cross_kv_bindings`).
    static_cross_kv: Vec<(String, Arc<Value>)>,
    context_tokens: Vec<TokenId>,
    /// Leading tokens whose KV was carried over from the previous generation and
    /// so must not be prefilled again.
    retained_len: usize,
    prompt_len: usize,
    generated_count: usize,
    /// Paged KV to mirror each step's `present.*` outputs into, when the
    /// decoder's KV can be paged.
    paged: Option<PagedMirror<'a>>,
    /// Tokens the decoder has actually run, and therefore the exact length its
    /// KV covers.
    ///
    /// Tracked rather than derived from `context_tokens`, because the two differ:
    /// `commit_token` appends the sampled token, but that token is not fed to the
    /// decoder until the *next* step, so at the end of a generation the context
    /// is one token longer than the KV. Retaining the context length would claim
    /// KV that does not exist and corrupt the next turn's attention.
    kv_len: usize,
}

impl PipelineDecodeLoopBackend<'_> {
    /// Run every declared `every_step` component over `seed` (the full prompt on
    /// prefill, the single running token on decode), publishing all of their
    /// outputs into the shared pool. Topological order ensures a component sees
    /// any upstream `every_step` output produced earlier in the same step.
    fn run_step_components(&mut self, seed: &[TokenId]) -> anyhow::Result<()> {
        if self.step_components.is_empty() {
            return Ok(());
        }
        let ids: Vec<i64> = seed.iter().map(|&t| i64::from(t)).collect();
        let seq = ids.len() as i64;
        for (binding, session) in &self.step_components {
            let mut inputs: Vec<(String, Value)> =
                Vec::with_capacity(binding.routed_inputs.len() + 1);
            for routed in &binding.routed_inputs {
                let value = self
                    .pool
                    .get(&routed.endpoint)
                    .or_else(|| {
                        routed
                            .routed_from
                            .as_deref()
                            .and_then(|from| self.pool.get(from))
                    })
                    .with_context(|| routed.missing_message.clone())?;
                inputs.push((
                    routed.port.clone(),
                    coerce_value_to_dtype(value, routed.dtype)?,
                ));
            }
            if let Some(port) = &binding.token_input {
                inputs.push((port.clone(), Value::from_slice_i64(&ids, &[1, seq])?));
            }
            let refs = inputs
                .iter()
                .map(|(name, value)| (name.as_str(), value))
                .collect::<Vec<_>>();
            let outputs = session.run(&refs).map_err(|e| {
                anyhow::anyhow!(
                    "ORT every_step component '{}' failed: {e}",
                    binding.component
                )
            })?;
            for (name, value) in session.output_names().iter().zip(outputs) {
                self.pool
                    .insert(format!("{}.{}", binding.component, name), value);
            }
        }
        Ok(())
    }

    /// Build this step's decoder extra inputs by re-reading every routed source
    /// endpoint from the shared pool. `every_step` outputs are already fresh
    /// (just re-run); cached `prompt_only` conditioning is simply re-read. The
    /// static encoder cross-attention KV (resolved once from the prologue) is
    /// appended verbatim so the decoder's `past_*_cross_%d` inputs are bound.
    fn decoder_extras(&self) -> anyhow::Result<Vec<(String, Value)>> {
        let mut extras =
            Vec::with_capacity(self.decoder_in_edges.len() + self.static_cross_kv.len());
        for (from, port) in &self.decoder_in_edges {
            let value = self.pool.get(from).with_context(|| {
                format!("missing routed pipeline tensor '{from}' for decoder input '{port}'")
            })?;
            extras.push((port.clone(), clone_value(value)?));
        }
        for (port, value) in &self.static_cross_kv {
            // The static cross-KV buffer is invariant across the decode loop, so
            // re-bind it as a no-copy alias over the shared owner instead of
            // deep-copying the (large) tensor every step.
            let aliased = Value::alias_with_shape(Arc::clone(value), value.shape())?;
            extras.push((port.clone(), aliased));
        }
        Ok(extras)
    }
}

impl DecodeLoopBackend for PipelineDecodeLoopBackend<'_> {
    fn context_len(&self) -> usize {
        self.context_tokens.len()
    }

    fn processor_prompt_tokens(&self) -> &[TokenId] {
        &self.context_tokens
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
        // On the first step feed only the tokens not already covered by
        // retained KV (`prompt_len` is the uncovered suffix, and equals the
        // whole prompt when nothing was retained); afterwards, the running token.
        let input_tokens = if self.decoder_state.use_kv && self.generated_count > 0 {
            self.context_tokens[self.context_tokens.len() - 1..].to_vec()
        } else {
            self.context_tokens[self.retained_len..].to_vec()
        };
        // Refresh every `every_step` component over exactly the tokens the
        // decoder is about to consume, then route their (and any cached) outputs
        // into the decoder for this step.
        self.run_step_components(&input_tokens)?;
        let extras = self.decoder_extras()?;
        let outputs = run_decode_step_with_extra(
            self.decoder,
            self.decoder_state,
            &input_tokens,
            past_len,
            &extras,
        )?;
        self.kv_len = past_len + input_tokens.len();
        // Mirror this step's KV into pages before the outputs are consumed, so
        // a later request opening with the same prefix can attach these pages
        // instead of recomputing them.
        if let Some(paged) = self.paged.as_mut().filter(|paged| !paged.exhausted) {
            // A windowed decoder's present tensor is indexed in *retained*
            // buffer space, not absolute position space: once the window has
            // evicted anything, an absolute index reads the wrong rows or runs
            // off the end. This is the same conversion the single-model decode
            // step does before mirroring.
            let retained_past_len = self.decoder_state.retained_kv_len(past_len);
            match mirror_present_kv_to_pages(
                self.decoder,
                paged.kv_model,
                paged.cache,
                paged.seq,
                &outputs,
                retained_past_len,
                input_tokens.len(),
            ) {
                Ok(()) => paged.mirrored_tokens = past_len + input_tokens.len(),
                // Mirroring exists so a *later* request can reuse this KV. The
                // pool running dry says nothing about whether this generation
                // is valid, so failing it would punish the caller for a cache
                // being full. Stop mirroring and keep decoding; only the
                // tokens already mirrored get published.
                Err(error) if is_kv_out_of_memory(&error) => {
                    paged.exhausted = true;
                    tracing::debug!(
                        "KV page pool exhausted after {} token(s); this generation stops \
                         publishing KV for reuse but continues normally ({error})",
                        paged.mirrored_tokens
                    );
                }
                Err(error) => return Err(error),
            }
            // Keep the paged sequence's window in step with the decoder's, so
            // the pages published for reuse describe what the decoder can
            // actually attend to.
            let pages_before = paged
                .cache
                .page_table
                .get_sequence(paged.seq)
                .map_or(0, <[_]>::len);
            apply_paged_sliding_window(
                paged.cache,
                paged.seq,
                self.decoder_state.sliding_window(),
                self.decoder_state.sink_tokens(),
            )?;
            let pages_after = paged
                .cache
                .page_table
                .get_sequence(paged.seq)
                .map_or(0, <[_]>::len);
            // Compared rather than inferred from the window size: only an
            // actual drop makes the sequence non-contiguous, so a windowed
            // model whose conversation still fits its window keeps publishing.
            if pages_after < pages_before {
                paged.windowed = true;
            }
        }
        extract_next_token_logits_with_io(
            self.decoder,
            outputs,
            self.decoder_state.io.logits_output.as_deref(),
        )
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
#[allow(clippy::large_enum_variant)] // Keep plans inline to preserve the existing allocation behavior.
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
    presence_conditions: HashMap<String, String>,
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
    /// Upstream components declared `every_step`: run on **every** autoregressive
    /// step — over the full expanded prompt during prefill and over the single
    /// running token during decode — in topological order, with all of their
    /// outputs routed into the decoder for that same step. This is the generic
    /// per-step component contract that replaces the former one-output
    /// `inputs_embeds` fusion special case: it refreshes every declared output
    /// (e.g. both `inputs_embeds` and a second sequence-dependent tensor), never
    /// just one, and never inspects tensor names to do so. Empty for a
    /// conventional text decoder.
    step_components: Vec<String>,
    /// Single-pass components declared `final_only`: run once, in declared
    /// order, after the decode loop completes (e.g. a TTS vocoder). Empty for a
    /// conventional text decoder or a Whisper-style ASR pipeline.
    post_decode_components: Vec<String>,
    dataflow: Vec<DataflowEdge>,
    presence_conditions: HashMap<String, String>,
}

/// Dual, hierarchically-nested autoregressive pipeline for the multi-decoder TTS
/// shape in DESIGN.md §20.3.
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
/// (`frame_codes [+ text_embed] -> inputs_embeds`). This keeps the engine generic:
/// the codec-sum construction lives in an ONNX component, not in Rust. See
/// [`PreEmbedderBinding`]. The inner loop is unchanged in both modes.
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
    /// Optional prefill-embedder binding, driving the talker's real frame-0
    /// PREFILL sequence and per-frame trailing-text conditioning. Only valid
    /// alongside `pre_embedder`. When `Some`, the driver looks up this
    /// prompt-phase component's pooled `prefill_output` / `trailing_output`
    /// tensors (all ports metadata-declared, see [`PrefillEmbedderBinding`]).
    /// When `None`, frame 0 uses a zero seed and every `text_embed` is zero
    /// (backward compatible).
    prefill_embedder: Option<PrefillEmbedderBinding>,
    dataflow: Vec<DataflowEdge>,
    presence_conditions: HashMap<String, String>,
}

/// Wiring for a pre-embedder that drives the outer talker's per-step
/// `inputs_embeds` in a [`NestedAutoregressivePlan`].
///
/// The talker consumes `inputs_embeds` (not `input_ids`), built each step from
/// the previous frame's codes as `codec_sum(+ text_embed)`. That construction is
/// materialized into an ONNX component with inputs `frame_codes [batch,
/// num_code_groups]` int64 (`[+ text_embed [batch, 1, hidden]]`) → output
/// `inputs_embeds [batch, 1, hidden]`. This binding records the component name
/// and the outer decoder input port fed by it; the exact pre-embedder input names
/// are resolved from its loaded session at drive time (sessions are not
/// available at plan-build time).
#[derive(Debug, Clone)]
struct PreEmbedderBinding {
    /// Pre-embedder component name (a declared model).
    component: String,
    /// Outer decoder input port that receives the per-step embeddings
    /// (`inputs_embeds`), from the required dataflow edge
    /// `{component}.{output_port} -> {outer}.inputs_embeds` (the edge `to` side).
    outer_input: String,
    /// Pre-embedder output port feeding the outer decoder, from the same edge's
    /// `from` side. Metadata-declared — never guessed by name/dtype.
    output_port: String,
    /// Pre-embedder input port receiving the previous frame's codes
    /// (`int64 [1, G]`). Metadata-declared via [`PreEmbedderSpec::frame_codes_input`].
    frame_codes_input: String,
    /// Optional pre-embedder input port receiving the per-frame trailing-text
    /// vector. Metadata-declared via [`PreEmbedderSpec::text_embed_input`].
    text_embed_input: Option<String>,
}

/// Wiring for the optional prefill embedder that supplies the outer talker's
/// frame-0 PREFILL sequence and per-frame trailing-text conditioning in a
/// [`NestedAutoregressivePlan`]. Every port is metadata-declared (from
/// [`PrefillEmbedderSpec`]); the runtime never guesses one by name or dtype.
#[derive(Debug, Clone)]
struct PrefillEmbedderBinding {
    /// Prefill-embedder component name (a declared, prompt-phase model).
    component: String,
    /// Input port receiving the tokenized prompt (`int64 [1, L]`).
    prompt_input: String,
    /// Output port carrying the talker's frame-0 PREFILL sequence
    /// (`float [1, prefill_len, hidden]`).
    prefill_output: String,
    /// Output port carrying the per-frame trailing-text vectors
    /// (`float [1, trailing_len, hidden]`).
    trailing_output: String,
}

/// One forward invocation of a single model with no runtime-managed loop.
#[derive(Debug, Clone)]
struct SinglePassPlan {
    model: String,
    /// Components that run once before the model (e.g. an encoder).
    prompt_components: Vec<String>,
    dataflow: Vec<DataflowEdge>,
    presence_conditions: HashMap<String, String>,
}

fn bind_dimension(
    dimensions: &mut HashMap<String, i64>,
    symbol: &str,
    value: i64,
    endpoint: &str,
) -> anyhow::Result<()> {
    if value < 0 {
        anyhow::bail!(
            "cannot resolve fallback shape symbol '{symbol}' for '{endpoint}' from dynamic \
             dimension {value}"
        );
    }
    if let Some(previous) = dimensions.insert(symbol.to_string(), value)
        && previous != value
    {
        anyhow::bail!(
            "conflicting values for fallback shape symbol '{symbol}': {previous} and {value} \
             while resolving '{endpoint}'"
        );
    }
    Ok(())
}

fn zero_value(shape: &[i64], dtype: DataType) -> anyhow::Result<Value> {
    let numel = shape.iter().try_fold(1usize, |count, &dimension| {
        let dimension = usize::try_from(dimension)
            .map_err(|_| anyhow::anyhow!("negative tensor dimension {dimension}"))?;
        count
            .checked_mul(dimension)
            .context("fallback tensor element count overflow")
    })?;
    match dtype {
        DataType::Float32 | DataType::Float16 | DataType::BFloat16 => {
            Value::from_f32_slice_as(&vec![0.0; numel], shape, dtype).map_err(Into::into)
        }
        DataType::Int64 => Value::from_slice_i64(&vec![0; numel], shape).map_err(Into::into),
        other => anyhow::bail!(
            "zero fallback materialization does not support graph input dtype {other:?}"
        ),
    }
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
        let path =
            std::path::Path::new(&dir).join(format!("step_{step:04}_{denoiser}_{port}.json"));
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
    presence_conditions: HashMap<String, String>,
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
        Value::from_slice_i64(&output, &shape)
            .map(Some)
            .map_err(Into::into)
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
        let batch = sequence_count
            .checked_div(sequence_length)
            .unwrap_or(0)
            .max(1);

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
                            let logit_row = &all_logits[position * vocab..(position + 1) * vocab];
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
        .wrapping_add(0xA076_1D64_78BD_642F);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    rng.random::<f64>()
}


/// DDIM (η = 0) noise schedule, precomputed per inference
/// step as `(alpha_cumprod_t, alpha_cumprod_prev)`.
///
/// Diffusion-standard update for a model that predicts noise `eps`:
///   `x0_hat = (x_t - sqrt(1 - a_t) * eps) / sqrt(a_t)`
///   `x_prev = sqrt(a_prev) * x0_hat + sqrt(1 - a_prev) * eps`
#[derive(Debug, Clone)]
struct DdimSchedule {
    steps: Vec<(f32, f32)>,
    timesteps: Vec<f32>,
    prediction: PredictionType,
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
            anyhow::bail!("scheduler num_steps ({num_steps}) must be in 1..={num_train_timesteps}");
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
        Ok(Self {
            steps,
            timesteps,
            prediction: PredictionType::Epsilon,
        })
    }

    /// Set the model parameterization (`epsilon` by default).
    fn with_prediction(mut self, prediction: PredictionType) -> Self {
        self.prediction = prediction;
        self
    }

    /// Apply one DDIM step to `sample` given the raw `model_out`. The model
    /// output is first converted to epsilon per [`Self::prediction`], then the
    /// epsilon-form DDIM update runs (byte-identical for `epsilon`).
    fn step(&self, k: usize, sample: &[f32], model_out: &[f32]) -> anyhow::Result<Vec<f32>> {
        if sample.len() != model_out.len() {
            anyhow::bail!(
                "scheduler sample/model_output length mismatch: {} vs {}",
                sample.len(),
                model_out.len()
            );
        }
        let (a_t, a_prev) = self.steps[k];
        let sqrt_a_t = a_t.sqrt();
        let sqrt_one_minus_a_t = (1.0 - a_t).sqrt();
        let sqrt_a_prev = a_prev.sqrt();
        let sqrt_one_minus_a_prev = (1.0 - a_prev).sqrt();
        Ok(sample
            .iter()
            .zip(model_out)
            .map(|(&x, &m)| {
                let e =
                    epsilon_from_model_output(m, x, sqrt_a_t, sqrt_one_minus_a_t, self.prediction);
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
        let stepped = DdimSchedule::step(
            self,
            step,
            &sample.to_vec_f32_lossy()?,
            &model_output.to_vec_f32_lossy()?,
        )?;
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
    /// Model parameterization (`epsilon` by default).
    prediction: PredictionType,
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
            anyhow::bail!("scheduler num_steps ({num_steps}) must be in 1..={num_train_timesteps}");
        }
        if let Some(sigmas) = spacing_sigmas(
            spacing,
            num_train_timesteps,
            beta_start,
            beta_end,
            beta_schedule,
            num_steps,
        )? {
            let train = training_sigmas(num_train_timesteps, beta_start, beta_end, beta_schedule)?;
            let timesteps = sigmas[..num_steps]
                .iter()
                .map(|&s| sigma_to_t(&train, s))
                .collect();
            return Ok(Self {
                sigmas,
                timesteps,
                prediction: PredictionType::Epsilon,
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
        let ts_denom = if num_steps > 1 {
            (num_steps - 1) as f32
        } else {
            1.0
        };
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
        Ok(Self {
            sigmas,
            timesteps,
            prediction: PredictionType::Epsilon,
        })
    }

    /// Set the model parameterization (`epsilon` by default).
    fn with_prediction(mut self, prediction: PredictionType) -> Self {
        self.prediction = prediction;
        self
    }

    /// `x / sqrt(sigma^2 + 1)` — scale the raw sample for the denoiser input.
    fn scale(&self, step: usize, sample: &[f32]) -> Vec<f32> {
        let factor = (self.sigmas[step] * self.sigmas[step] + 1.0).sqrt();
        sample.iter().map(|&x| x / factor).collect()
    }

    /// `x_next = x + eps * (sigma_next - sigma)` on the raw sample. The raw
    /// `model_out` is first converted to the epsilon derivative per
    /// [`Self::prediction`] (byte-identical for `epsilon`).
    fn step_vec(&self, step: usize, sample: &[f32], model_out: &[f32]) -> anyhow::Result<Vec<f32>> {
        if sample.len() != model_out.len() {
            anyhow::bail!(
                "scheduler sample/model_output length mismatch: {} vs {}",
                sample.len(),
                model_out.len()
            );
        }
        let sigma = self.sigmas[step];
        // Convert to DDPM alpha/sigma: alpha_t = 1/sqrt(sigma^2+1),
        // sigma_t = sigma * alpha_t, and the DDPM latent x_t = alpha_t * sample.
        // The epsilon derivative diffusers feeds is `(sample - x0) / sigma`,
        // which equals the DDPM epsilon; it reduces to `model_out` for epsilon.
        let alpha_t = 1.0 / (sigma * sigma + 1.0).sqrt();
        let sigma_t = sigma * alpha_t;
        let dt = self.sigmas[step + 1] - self.sigmas[step];
        Ok(sample
            .iter()
            .zip(model_out)
            .map(|(&x, &m)| {
                let e =
                    epsilon_from_model_output(m, alpha_t * x, alpha_t, sigma_t, self.prediction);
                x + e * dt
            })
            .collect())
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
    prediction: PredictionType,
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
        Ok(Self {
            sigmas: euler.sigmas,
            timesteps: euler.timesteps,
            prediction: PredictionType::Epsilon,
        })
    }

    /// Set the model parameterization (`epsilon` by default).
    fn with_prediction(mut self, prediction: PredictionType) -> Self {
        self.prediction = prediction;
        self
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
        let model_out = model_output.to_vec_f32_lossy()?;
        let sigma_from = self.sigmas[step];
        let sigma_to = self.sigmas[step + 1];
        let sigma_up = (sigma_to * sigma_to * (sigma_from * sigma_from - sigma_to * sigma_to)
            / (sigma_from * sigma_from))
            .max(0.0)
            .sqrt();
        let sigma_down = (sigma_to * sigma_to - sigma_up * sigma_up).max(0.0).sqrt();
        let dt = sigma_down - sigma_from;
        // DDPM alpha/sigma for the current sigma; DDPM latent x_t = alpha_t * x.
        let alpha_t = 1.0 / (sigma_from * sigma_from + 1.0).sqrt();
        let sigma_t = sigma_from * alpha_t;
        let noise = noise
            .context("euler_ancestral requires per-step noise")?
            .to_vec_f32_lossy()?;
        if noise.len() != x.len() {
            anyhow::bail!(
                "euler_ancestral noise length {} != sample {}",
                noise.len(),
                x.len()
            );
        }
        let out: Vec<f32> = (0..x.len())
            .map(|i| {
                let e = epsilon_from_model_output(
                    model_out[i],
                    alpha_t * x[i],
                    alpha_t,
                    sigma_t,
                    self.prediction,
                );
                x[i] + e * dt + noise[i] * sigma_up
            })
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
        let scaled: Vec<f32> = sample
            .to_vec_f32_lossy()?
            .iter()
            .map(|&x| x / factor)
            .collect();
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
    /// Model parameterization (`epsilon` by default).
    prediction: PredictionType,
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
            anyhow::bail!("scheduler num_steps ({num_steps}) must be in 1..={num_train_timesteps}");
        }
        if let Some(sigmas) = spacing_sigmas(
            spacing,
            num_train_timesteps,
            beta_start,
            beta_end,
            beta_schedule,
            num_steps,
        )? {
            let train = training_sigmas(num_train_timesteps, beta_start, beta_end, beta_schedule)?;
            let timesteps = sigmas[..num_steps]
                .iter()
                .map(|&s| sigma_to_t(&train, s))
                .collect();
            return Ok(Self {
                sigmas,
                timesteps,
                prev_x0: Mutex::new(None),
                prediction: PredictionType::Epsilon,
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
            prediction: PredictionType::Epsilon,
        })
    }

    /// Set the model parameterization (`epsilon` by default).
    fn with_prediction(mut self, prediction: PredictionType) -> Self {
        self.prediction = prediction;
        self
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
        let model_out = model_output.to_vec_f32_lossy()?;
        if x.len() != model_out.len() {
            anyhow::bail!(
                "dpm++ sample/model_output length mismatch: {} vs {}",
                x.len(),
                model_out.len()
            );
        }

        let sigma = self.sigmas[step];
        let (alpha_t0, sigma_t0) = dpm_alpha_sigma(sigma);
        // Data prediction (x0) from the raw model output per the parameterization.
        // For epsilon this is the byte-identical (x - sigma_t*eps)/alpha_t.
        let x0: Vec<f32> = x
            .iter()
            .zip(&model_out)
            .map(|(&xi, &mi)| x0_from_model_output(mi, xi, alpha_t0, sigma_t0, self.prediction))
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
        let step_components = step_phase_components(spec, &decoder)?;
        let post_decode_components = post_decode_components(spec, &decoder)?;
        Ok(Self::Autoregressive(AutoregressivePlan {
            decoder,
            prompt_components,
            step_components,
            post_decode_components,
            dataflow: spec.dataflow.clone(),
            presence_conditions: presence_conditions(spec),
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
        let pre_embedder = match nested.pre_embedder.as_ref() {
            Some(spec_pre) => {
                let name = spec_pre.component.as_str();
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
                             dataflow edge '{name}.<output> -> {outer}.inputs_embeds'"
                        )
                    })?;
                // Both ports come from the REQUIRED edge (metadata): the outer
                // decoder input from the `to` side and the pre-embedder output
                // from the `from` side. The `frame_codes` / `text_embed` inputs
                // come from the PreEmbedderSpec. Nothing is guessed by name/dtype.
                let (_, outer_input) = parse_endpoint(&edge.to)?;
                let (_, output_port) = parse_endpoint(&edge.from)?;
                Some(PreEmbedderBinding {
                    component: name.to_string(),
                    outer_input: outer_input.to_string(),
                    output_port: output_port.to_string(),
                    frame_codes_input: spec_pre.frame_codes_input.clone(),
                    text_embed_input: spec_pre.text_embed_input.clone(),
                })
            }
            None => None,
        };
        let pre_embedder_component = pre_embedder.as_ref().map(|p| p.component.clone());

        // Optional prefill-embedder: supplies the talker's real frame-0 PREFILL
        // sequence (`prefill_embeds`) and per-frame trailing-text conditioning
        // (`trailing_text_embeds`), both materialized from the tokenized prompt.
        // It runs as an ordinary prompt-phase component (`run_on: prompt_only` /
        // `on_demand`); the driver reads its pooled outputs after the prompt
        // phase. It must be a declared model distinct from the loop decoders and
        // the pre-embedder, and only makes sense alongside a `pre_embedder` (the
        // trailing-text vectors are threaded through it on frames >= 1).
        let prefill_embedder = match nested.prefill_embedder.as_ref() {
            Some(spec_prefill) => {
                let name = spec_prefill.component.as_str();
                if !spec.models.contains_key(name) {
                    anyhow::bail!(
                        "nested_autoregressive prefill_embedder '{name}' is not declared in models"
                    );
                }
                if name == outer || name == inner {
                    anyhow::bail!(
                        "nested_autoregressive prefill_embedder '{name}' must be distinct from \
                         the outer/inner decoders"
                    );
                }
                if pre_embedder_component.as_deref() == Some(name) {
                    anyhow::bail!(
                        "nested_autoregressive prefill_embedder '{name}' must be distinct from \
                         the pre_embedder"
                    );
                }
                if pre_embedder.is_none() {
                    anyhow::bail!(
                        "nested_autoregressive prefill_embedder '{name}' requires a 'pre_embedder' \
                         (frames >= 1 thread its trailing-text vectors through the pre-embedder)"
                    );
                }
                // All ports (prompt input, prefill and trailing outputs) are
                // metadata-declared in the PrefillEmbedderSpec — never guessed.
                Some(PrefillEmbedderBinding {
                    component: name.to_string(),
                    prompt_input: spec_prefill.prompt_input.clone(),
                    prefill_output: spec_prefill.prefill_output.clone(),
                    trailing_output: spec_prefill.trailing_output.clone(),
                })
            }
            None => None,
        };

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
            prefill_embedder,
            dataflow: spec.dataflow.clone(),
            presence_conditions: presence_conditions(spec),
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
            presence_conditions: presence_conditions(spec),
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
                        format!(
                            "composite stage '{}' (single_pass) is missing 'model'",
                            stage.name
                        )
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
            presence_conditions: presence_conditions(spec),
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
            scheduler: build_scheduler(
                spec.strategy.scheduler_config.as_ref(),
                num_steps,
                schedulers,
            )?,
            cfg_conditioning_input: spec.strategy.cfg_conditioning_input.clone(),
            dataflow: spec.dataflow.clone(),
            scheduler_spec: spec.strategy.scheduler_config.clone(),
            scheduler_registry: schedulers.clone(),
            presence_conditions: presence_conditions(spec),
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

    fn presence_condition(&self, component: &str) -> Option<&str> {
        let conditions = match self {
            Self::Autoregressive(plan) => &plan.presence_conditions,
            Self::NestedAutoregressive(plan) => &plan.presence_conditions,
            Self::SinglePass(plan) => &plan.presence_conditions,
            Self::Iterative(plan) => &plan.presence_conditions,
            Self::Composite(plan) => &plan.presence_conditions,
        };
        conditions.get(component).map(String::as_str)
    }

    fn component_is_present(&self, component: &str, present: &BTreeSet<String>) -> bool {
        self.presence_condition(component)
            .is_none_or(|key| present.contains(key))
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

fn presence_conditions(spec: &PipelineSpec) -> HashMap<String, String> {
    spec.phases
        .iter()
        .filter_map(|(component, phase)| {
            phase
                .when_present
                .as_ref()
                .map(|key| (component.clone(), key.clone()))
        })
        .collect()
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

/// Collect the `every_step`-phase components (excluding the `decoder` itself) in
/// topological order.
///
/// These upstream components run on **every** autoregressive step: over the full
/// expanded prompt during prefill and over the single running token during
/// decode. Their outputs are routed into the decoder for that same step, so a
/// component that emits several sequence-dependent tensors (for example both
/// `inputs_embeds` and a per-layer conditioning tensor) has all of them
/// refreshed generically — the engine never special-cases a single output or
/// infers roles from tensor names. Topological order lets one `every_step`
/// component consume an earlier one's freshly produced output within the step.
fn step_phase_components(spec: &PipelineSpec, decoder: &str) -> anyhow::Result<Vec<String>> {
    let mut step = Vec::new();
    for component in topological_components(spec)? {
        if component == decoder {
            continue;
        }
        if let PhaseRunOn::EveryStep = component_phase(spec, &component, decoder) {
            step.push(component);
        }
    }
    Ok(step)
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
    /// Pre-embedder output port feeding the outer decoder. Metadata-declared
    /// (from the required dataflow edge's `from` side) — never guessed.
    output_port: String,
    /// Pre-embedder input receiving the previous frame's codes (int64 `[1, G]`).
    /// Metadata-declared via `PreEmbedderSpec::frame_codes_input`.
    frame_codes_input: String,
    /// Optional trailing-text input. Fed the prefill embedder's per-frame
    /// `trailing_text_embeds` slice when a `prefill_embedder` is set, else zeros.
    /// Metadata-declared via `PreEmbedderSpec::text_embed_input`.
    text_embed_input: Option<String>,
    /// Embedding hidden size for the emitted `inputs_embeds` / `text_embed`.
    hidden: usize,
}

/// The optional prefill embedder's resolved, pooled outputs: the talker's
/// frame-0 multi-position PREFILL sequence and the per-frame trailing-text
/// vectors consumed as the pre-embedder's `text_embed` on frames `k >= 1`.
struct ResolvedPrefill {
    /// `prefill_embeds` [1, prefill_len, hidden]: the talker's frame-0 seed.
    prefill_embeds: Value,
    /// Number of prefill positions (`prefill_embeds.shape()[1]`).
    prefill_len: usize,
    /// Flattened `trailing_text_embeds` [1, trailing_len, hidden] as row-major
    /// f32 (`trailing[i*hidden..(i+1)*hidden]` is the vector for frame `i + 1`).
    trailing: Vec<f32>,
    /// Number of trailing-text vectors (`trailing_text_embeds.shape()[1]`).
    trailing_len: usize,
    /// Embedding hidden size (matches the pre-embedder's `hidden`).
    hidden: usize,
}

/// Build the outer talker's per-step `inputs_embeds` by running the codec-sum
/// pre-embedder over one frame's `frame_codes` (`[outer_code_0, inner_code_1,
/// ..., inner_code_{G-1}]`). Returns a `[1, 1, hidden]` embedding.
///
/// When `text_embed` is `Some`, that `[hidden]` slice is fed as the
/// trailing-text conditioning input (the prefill embedder's per-frame
/// `trailing_text_embeds` vector). When `None`, a zero `[1, 1, hidden]` tensor is
/// fed (the backward-compatible no-prefill_embedder path).
///
/// Every port used here (`frame_codes_input`, `text_embed_input`, `output_port`)
/// is metadata-declared on [`ResolvedPreEmbedder`] — there is NO name/dtype
/// guessing of the pre-embedder's ports.
fn run_pre_embedder(
    pre: &ResolvedPreEmbedder<'_>,
    frame_codes: &[i64],
    text_embed: Option<&[f32]>,
) -> anyhow::Result<Value> {
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
        let data = match text_embed {
            Some(slice) => slice.to_vec(),
            None => vec![0.0f32; pre.hidden],
        };
        inputs.push((
            name.clone(),
            Value::from_f32_slice_as(&data, &[1, 1, pre.hidden as i64], dtype)
                .map_err(|e| anyhow::anyhow!("failed to build text_embed: {e}"))?,
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
    // Select the metadata-declared output port (never guessed by name).
    let index = pre
        .session
        .output_names()
        .iter()
        .position(|name| name == &pre.output_port)
        .with_context(|| {
            format!(
                "pre-embedder has no declared output port '{}'",
                pre.output_port
            )
        })?;
    let value = outputs
        .get(index)
        .context("pre-embedder produced no output for its declared port")?;
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
        assert!(
            (n1[0] - (std::f32::consts::SQRT_2 - 1.0)).abs() < 1e-5,
            "{}",
            n1[0]
        );
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
        let sched =
            Dpmpp2m::with_schedule(num_train, 0.00085, 0.012, "scaled_linear", num_steps, "")
                .expect("schedule builds");
        let timesteps = sched.timesteps().expect("dpm++ exposes timesteps");
        let denom = (num_train - 1) as f32;
        let mut expected: Vec<f32> = (0..=num_steps)
            .map(|j| (j as f32 * denom / num_steps as f32).round_ties_even())
            .collect();
        expected.reverse();
        expected.pop();
        assert_eq!(timesteps.len(), num_steps);
        assert!(
            (timesteps[0] - 999.0).abs() < 1e-3,
            "first timestep {}",
            timesteps[0]
        );
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
                value = sched
                    .step(step, num_steps, &value, &logits_value)
                    .expect("step");
            }
            value.to_vec_i64().expect("tokens")
        };

        let out = run(&sched);
        assert_eq!(&out[..prompt_len], &[1, 2], "prompt prefix preserved");
        for (pos, &tok) in out.iter().enumerate() {
            assert_ne!(
                tok, mask_id,
                "position {pos} still masked / emitted the mask token"
            );
        }
        for (offset, &token) in out[prompt_len..seq].iter().enumerate() {
            let pos = prompt_len + offset;
            assert_eq!(token, (pos % 4) as i64, "position {pos} token");
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
            sample
                .to_vec_f32()
                .unwrap()
                .iter()
                .all(|value| value.is_finite()),
            "final dpm++ sample must be finite"
        );
    }

    fn component(role: &str) -> PipelineComponentSpec {
        PipelineComponentSpec {
            filename: format!("{role}.onnx"),
            role: role.to_string(),
            device_preference: None,
            tokenizer: None,
            io: None,
        }
    }

    #[cfg(not(feature = "native-backend"))]
    #[test]
    fn explicit_native_backend_without_feature_reports_actionable_build_error() {
        let error = PipelineEngine::from_dir_with_config(
            Path::new("does-not-need-to-exist"),
            EngineConfig {
                decode_backend: EngineDecodeBackend::Native,
                ..EngineConfig::default()
            },
        )
        .err()
        .expect("native pipeline backend must report an actionable error");
        let message = error.to_string();
        assert!(
            message.contains("without the 'native-backend' feature"),
            "unexpected error: {message}"
        );
        // No longer the blanket "not supported for pipeline models" rejection.
        assert!(!message.contains("native backend not supported for pipeline models"));
    }

    #[cfg(feature = "native-backend")]
    #[test]
    fn auto_backend_routes_native_only_pipeline_to_the_native_backend() -> anyhow::Result<()> {
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
            .expect("Auto must engage the native backend for native-only pipeline components");
        let message = error.to_string();
        // Auto now selects the native backend and constructs components through
        // the backend-neutral seam rather than emitting the old blanket refusal.
        assert!(
            message.contains("native"),
            "error should reference the native backend: {message}"
        );
        assert!(!message.contains("native backend not supported for pipeline models"));
        Ok(())
    }

    #[test]
    fn plan_routes_prompt_encoder_outputs_to_decoder_inputs() -> anyhow::Result<()> {
        let spec = PipelineSpec {
            audio: None,
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
                prefill_embedder: None,
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
                            prefill_embedder: None,
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
                            prefill_embedder: None,
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
                        when_present: None,
                    },
                ),
                (
                    "decoder".to_string(),
                    PhaseConfig {
                        run_on: PhaseRunOn::EveryStep,
                        when_present: None,
                    },
                ),
            ]),
            vision: None,
            positions: None,
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
            prefill_embedder: None,
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
            audio: None,
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
            positions: None,
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
            audio: None,
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
            positions: None,
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
            ..Default::default()
        }
    }

    #[test]
    fn image_placeholder_expansion_replaces_tokens() {
        // [1, PLACEHOLDER, 2] with 2 tiles × 3 tokens/tile → [1, IMG, IMG, IMG, IMG, IMG, IMG, 2]
        let tokens: Vec<TokenId> = vec![1, 100, 2];
        let cfg = vision_config(100, 3);
        let expanded = expand_image_placeholders_count_based(tokens, Some(2), Some(&cfg)).unwrap();
        assert_eq!(expanded, vec![1, 100, 100, 100, 100, 100, 100, 2]);
    }

    #[test]
    fn image_placeholder_expansion_multiple_placeholders_errors() {
        // Count-based path only supports a single placeholder; >1 must error.
        let tokens: Vec<TokenId> = vec![100, 5, 100];
        let cfg = vision_config(100, 4);
        let err = expand_image_placeholders_count_based(tokens, Some(1), Some(&cfg)).unwrap_err();
        assert!(
            err.to_string()
                .contains("multi-image count-based expansion is not supported"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn image_placeholder_expansion_none_tiles_is_noop() {
        let tokens: Vec<TokenId> = vec![1, 100, 2];
        let cfg = vision_config(100, 256);
        let result =
            expand_image_placeholders_count_based(tokens.clone(), None, Some(&cfg)).unwrap();
        assert_eq!(result, tokens);
    }

    #[test]
    fn image_placeholder_expansion_no_vision_config_with_tiles_errors() {
        let tokens: Vec<TokenId> = vec![1, 100, 2];
        let err = expand_image_placeholders_count_based(tokens, Some(1), None).unwrap_err();
        assert!(err.to_string().contains("no vision section"));
    }

    #[test]
    fn image_placeholder_expansion_incomplete_contract_errors() {
        let tokens: Vec<TokenId> = vec![1, 100, 2];
        let cfg = PipelineVisionConfig {
            image_placeholder_token_id: Some(100),
            tokens_per_tile: None,
            ..Default::default()
        };
        let err = expand_image_placeholders_count_based(tokens, Some(1), Some(&cfg)).unwrap_err();
        assert!(err.to_string().contains("vision contract is incomplete"));
    }

    #[test]
    fn image_placeholder_expansion_missing_placeholder_errors() {
        let tokens: Vec<TokenId> = vec![1, 2, 3];
        let cfg = vision_config(100, 4);
        let err = expand_image_placeholders_count_based(tokens, Some(1), Some(&cfg)).unwrap_err();
        assert!(err.to_string().contains("no image placeholder token"));
    }

    #[test]
    fn image_placeholder_expansion_negative_id_errors() {
        let tokens: Vec<TokenId> = vec![1, 2];
        let cfg = PipelineVisionConfig {
            image_placeholder_token_id: Some(-1),
            tokens_per_tile: Some(4),
            ..Default::default()
        };
        let err = expand_image_placeholders_count_based(tokens, Some(1), Some(&cfg)).unwrap_err();
        assert!(err.to_string().contains("out of range"));
    }

    #[test]
    fn image_placeholder_expansion_tokens_per_tile_zero_errors() {
        let tokens: Vec<TokenId> = vec![1, 100, 2];
        let cfg = vision_config(100, 0);
        let err = expand_image_placeholders_count_based(tokens, Some(1), Some(&cfg)).unwrap_err();
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
        let err = expand_image_placeholders_count_based(tokens, Some(0), Some(&cfg)).unwrap_err();
        assert!(
            err.to_string().contains("empty token sequence"),
            "unexpected error: {err}"
        );
    }

    /// The reuse key must separate requests that compute different things, not
    /// just requests with different attachments.
    mod request_identity {
        use super::super::*;

        fn request() -> PipelineGenerateRequest {
            PipelineGenerateRequest::new(GenerateRequest::new(GeneratePrompt::TokenIds(vec![1, 2])))
        }

        fn pixels(value: f32) -> Value {
            Value::from_slice_f32(&[value; 4], &[1, 4]).expect("build a test tensor")
        }

        #[test]
        fn identical_requests_share_a_key() {
            let left = request().with_input("encoder.pixel_values", pixels(1.0));
            let right = request().with_input("encoder.pixel_values", pixels(1.0));
            assert_eq!(
                PipelineEngine::digest_request_identity(&left),
                PipelineEngine::digest_request_identity(&right)
            );
        }

        #[test]
        fn a_different_attachment_changes_the_key() {
            let left = request().with_input("encoder.pixel_values", pixels(1.0));
            let right = request().with_input("encoder.pixel_values", pixels(2.0));
            assert_ne!(
                PipelineEngine::digest_request_identity(&left),
                PipelineEngine::digest_request_identity(&right)
            );
        }

        #[test]
        fn presence_keys_change_the_key() {
            // Presence gates which components run and which optional decoder
            // inputs are bound, so the same tensors under different presence
            // keys describe a different computation.
            let left = request().with_input("encoder.pixel_values", pixels(1.0));
            let right = request()
                .with_input("encoder.pixel_values", pixels(1.0))
                .with_presence("audio");
            assert_ne!(
                PipelineEngine::digest_request_identity(&left),
                PipelineEngine::digest_request_identity(&right)
            );
        }

        #[test]
        fn the_image_tile_count_changes_the_key() {
            let left = request().with_input("encoder.pixel_values", pixels(1.0));
            let right = request()
                .with_input("encoder.pixel_values", pixels(1.0))
                .with_image_tile_count(4);
            assert_ne!(
                PipelineEngine::digest_request_identity(&left),
                PipelineEngine::digest_request_identity(&right)
            );
        }

        #[test]
        fn an_undigestible_request_disables_reuse() {
            // Nothing bound is still a well-defined identity; the `None` case is
            // reserved for tensors whose bytes cannot be read.
            assert!(PipelineEngine::digest_request_identity(&request()).is_some());
        }
    }
}

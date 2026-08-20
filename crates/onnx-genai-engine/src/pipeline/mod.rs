//! Multi-model pipeline orchestrator.

use crate::MemoryStrategyPlan;
use crate::decode::{
    DecodeState, apply_paged_sliding_window, clone_value, run_decode_step_with_extra,
};
use crate::decode_loop::{DecodeLoopBackend, DecodeLoopState, run_decode_loop};
use crate::engine::{
    Engine, EngineConfig, EngineResourceGovernor, MemoryStrategyPlanInput, analyze_model_memory,
    build_memory_strategy_plan, combine_graph_memory, component_governor, log_memory_strategy_plan,
    model_requires_native_backend, requested_decode_backend,
    resolve_memory_strategy_hot_tier_bytes, resolve_vram_limit_bytes,
};
use crate::kv_bridge::{
    KvModelInfo, attach_pages_to_sequence, infer_kv_model_info, load_materialized_past,
    sequence_pages_for_len,
};
use crate::logits::TokenId;
use crate::memory_authority::{MemoryAuthorityProvider, SharedMemoryAuthorityProvider};

use crate::pipeline_cache::{
    ComponentOutputCache, Digest, DigestBuilder, PREFIX_KEY_PREAMBLE, PipelineCacheStats,
    RetainedContext, absorb_value, digest_named_values, graph_is_deterministic, prefix_key,
};
use crate::processors::build_processor_chain;
use crate::{
    EngineDecodeBackend, GenerateOptions, GeneratePrompt, GenerateRequest, GenerateResult,
    GenerateTokenCallback, StopSequence,
};
use anyhow::Context;
use onnx_genai_kv::{PagedKvCache, PrefixCache, SequenceId};
use onnx_genai_metadata::{
    AbsentInputKind, DataflowEdge, ModelIoSpec, PhaseRunOn, PipelineSpec, PipelineStrategy,
    PipelineStrategyKind, PipelineVisionConfig, SchedulerSpec, SequenceInputKind, TensorDimension,
};
use onnx_genai_ort::{
    DataType, PipelineModelDirectory, PipelineModels, Session, SessionOptions, Tokenizer, Value,
};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;
use std::sync::Arc;

mod decoder_component;
mod flat_autoregressive;
mod iterative;
mod nested_autoregressive;
mod paged_decode;
mod prefix_reuse;
mod routing;
pub use routing::is_missing_required_input;
mod schedulers;
#[cfg(feature = "native-backend")]
pub(crate) use decoder_component::NativePipelineDecoder;
pub(crate) use decoder_component::{OrtPipelineDecoder, PipelineDecoderComponent};
pub use schedulers::{Scheduler, SchedulerFactory, SchedulerRegistry};

/// Named tensors supplied to or produced by pipeline components.
///
/// Keys are fully-qualified endpoints of the form `component.input_name` or
/// `component.output_name`.
pub type PipelineTensors = HashMap<String, Value>;

/// A typed request for an iterative image pipeline.
///
/// `pipeline` carries the package-specific prompt embeddings, seed latent, and
/// optional scheduler overrides.  Keeping those inputs in the existing generic
/// request lets model packages declare their own ports while this API owns the
/// image-specific result and latent-stream contract.
pub struct ImageRequest {
    pub pipeline: PipelineGenerateRequest,
    /// Fully-qualified final image endpoint. When omitted, the sole output of
    /// the final pipeline component is used.
    pub image_output: Option<String>,
}

impl ImageRequest {
    pub fn new(pipeline: PipelineGenerateRequest) -> Self {
        Self {
            pipeline,
            image_output: None,
        }
    }

    pub fn with_image_output(mut self, endpoint: impl Into<String>) -> Self {
        self.image_output = Some(endpoint.into());
        self
    }
}

impl From<PipelineGenerateRequest> for ImageRequest {
    fn from(pipeline: PipelineGenerateRequest) -> Self {
        Self::new(pipeline)
    }
}

/// The loop-carried latents after one denoising step.
pub struct ImageStep {
    /// The zero-based scheduler step.
    pub step: usize,
    /// Values keyed by their fully-qualified denoiser input endpoints.
    pub latents: PipelineTensors,
}

/// A fallible observer invoked immediately after every scheduler update.
pub type ImageStepCallback<'a> = dyn FnMut(&ImageStep) -> anyhow::Result<()> + 'a;

/// Typed final image result from an iterative pipeline.
pub struct ImageOutput {
    /// The image tensor selected from the final pipeline stage.
    pub image: Value,
    /// The final loop-carried latents, keyed by denoiser input endpoint.
    pub latents: PipelineTensors,
}

/// Stepwise image-generation result.
///
/// Rust does not yet have stable generator traits suitable for borrowing a
/// mutable pipeline engine, so [`PipelineEngine::generate_image`] records the
/// same per-step events exposed live by
/// [`PipelineEngine::generate_image_with_callback`].
pub struct ImageStream {
    pub steps: Vec<ImageStep>,
    pub output: ImageOutput,
}

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
    /// Owns the pipeline's component-weight reservation. It is created before
    /// any component session, so a failed load rolls the reservation back.
    resource_governor: EngineResourceGovernor,
    /// Autoregressive pipelines need a second plan after session graph I/O is
    /// available. Both governors delegate device accounting to the same
    /// authority; this one owns only the KV plan.
    _kv_governor: Option<EngineResourceGovernor>,
    memory_strategy_plan: MemoryStrategyPlan,
    plan: PipelinePlan,
    decode_backend: EngineDecodeBackend,
    generation_defaults: Option<onnx_genai_metadata::GenerationDefaults>,
    max_sequence_length: Option<usize>,
    eos_token_ids: Vec<TokenId>,
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
    /// Native decoder retained across sequential requests. Its CUDA KV bindings
    /// stay device-resident in their original dtype (including BF16); `retained`
    /// identifies the exact token prefix those bindings contain.
    native_retained_decoder: Option<Box<dyn PipelineDecoderComponent>>,
    /// Maximum suffix tokens sent through one native prefill graph invocation.
    /// Declared by `model.runtime_configurable.chunked_prefill`.
    prefill_chunk_size: Option<usize>,
    /// Paged KV for the decoder, when its `present.*` outputs describe a layout
    /// the page table can address.
    paged: Option<PipelinePagedKv>,
    /// Decoder device requested via [`EngineConfig::native_device`] (e.g. from
    /// `--ep cuda`), honored by the native pipeline decoder when the
    /// `ONNX_GENAI_PIPELINE_NATIVE_DECODER_DEVICE` override is unset.
    native_device: Option<crate::native_decode_device::NativeDecodeDevice>,
    /// Prompt-phase / post-decode component sessions loaded on the native
    /// backend, built lazily by the prologue the first time an active component
    /// with no ORT session runs. Empty on the ORT backend and for native
    /// pipelines whose prompt components never activate (e.g. a text-only prompt
    /// through a multimodal package never builds the vision encoder). Behind a
    /// `RefCell` because the prologue runs from `&self` paths.
    #[cfg_attr(
        not(feature = "native-backend"),
        allow(
            dead_code,
            reason = "native prompt sessions are built only by the native backend"
        )
    )]
    native_prompt_sessions:
        RefCell<BTreeMap<String, Box<dyn onnx_genai_metadata::ComponentSession>>>,
    #[cfg(feature = "cuda")]
    native_cuda_authority: Option<crate::memory_authority::DeviceMemoryAuthority>,
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

/// Resolve and validate an explicitly requested pipeline backend without
/// touching model files. Server construction calls this before its own package
/// discovery so both entry points preserve the same fail-fast behavior.
pub fn validate_pipeline_backend_request(
    requested: EngineDecodeBackend,
) -> anyhow::Result<EngineDecodeBackend> {
    let backend = requested_decode_backend(requested)?;
    #[cfg(not(feature = "native-backend"))]
    if backend == EngineDecodeBackend::Native {
        return Err(native_backend_not_compiled_error());
    }
    Ok(backend)
}

/// Construct all pipeline components through the native
/// [`ComponentSession`](onnx_genai_metadata::ComponentSession) seam, then report
/// that native decode does not yet cover this pipeline's (non-flat-autoregressive)
/// strategy.
///
/// GAP-3 Inc-A wires native decode for the **flat autoregressive** plan by driving
/// every component through the backend-neutral component / decoder builders inside
/// the shared decode loop. Other strategies (nested autoregressive TTS, iterative
/// diffusion, single-pass, composite) are not wired yet, so this returns a clear,
/// actionable error naming the loaded components and directing the caller to the
/// ORT backend, rather than a blanket "native backend not supported" rejection.
#[cfg(feature = "native-backend")]
fn native_pipeline_plan_unsupported(
    directory: &PipelineModelDirectory,
    config: &EngineConfig,
) -> anyhow::Error {
    let components = match build_native_pipeline_components(directory, config) {
        Ok(components) => components,
        Err(err) => return err,
    };
    let component_list = components.keys().cloned().collect::<Vec<_>>().join(", ");
    anyhow::anyhow!(
        "native pipeline decode currently supports only flat autoregressive pipelines \
         (GAP-3 Inc-A). All {} pipeline component(s) loaded successfully on the native backend \
         and expose their graph I/O through the backend-neutral component-session interface \
         (components: {}), but this pipeline uses a non-autoregressive strategy (nested \
         autoregressive, iterative/diffusion, single-pass, or composite) whose native decode \
         path is not wired yet. To run this pipeline today, set decode_backend = \
         EngineDecodeBackend::Ort (or ONNX_GENAI_BACKEND=ort).",
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

    // Deliberately CPU-neutral: this path exists only to name the components in
    // an unsupported-plan error, so it must not claim device memory to do it.
    let device = crate::engine::resolve_native_decode_device(
        config.native_device.clone(),
        &SessionOptions::default(),
    )?;
    let mut components: std::collections::BTreeMap<String, Box<dyn ComponentSession>> =
        std::collections::BTreeMap::new();
    for (name, path) in &directory.model_paths {
        let session = NativeComponentSession::load(
            path,
            device.clone(),
            crate::native_component::ComponentMemory::SelfProvisioned(None),
        )
        .with_context(|| {
            format!("failed to construct pipeline component '{name}' on the native backend")
        })?;
        components.insert(name.clone(), Box::new(session));
    }
    Ok(components)
}

/// Every_step components the operator explicitly requested be run on the native
/// nxrt backend, from `ONNX_GENAI_PIPELINE_NATIVE_STEP_COMPONENTS` (a
/// comma-separated list of component names). Empty/unset means all every_step
/// components run on the default ORT backend, so the ORT decode path is
/// unchanged. This is the injection seam for the native multi-component inc1
/// hybrid: it drives named every_step components natively inside an
/// otherwise-ORT pipeline decode loop, proving the value-type seam.
fn native_step_component_set() -> BTreeSet<String> {
    std::env::var("ONNX_GENAI_PIPELINE_NATIVE_STEP_COMPONENTS")
        .ok()
        .map(|list| {
            list.split(',')
                .map(str::trim)
                .filter(|name| !name.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Whether the flat autoregressive pipeline should drive `decoder` through the
/// native nxrt backend instead of ONNX Runtime, from
/// `ONNX_GENAI_PIPELINE_NATIVE_DECODER`. The value may name the decoder component
/// exactly, or be a truthy token (`1`/`true`/`yes`/`on`/`all`) selecting whatever
/// decoder the pipeline declares. Unset/empty keeps the ORT decoder (default, so
/// the ORT decode path is unchanged). This mirrors the Inc1 every_step selection
/// flag `ONNX_GENAI_PIPELINE_NATIVE_STEP_COMPONENTS`: it is the injection seam for
/// the native device-KV decoder (inc2b), driving the decoder natively inside an
/// otherwise-ORT pipeline decode loop while keeping its KV session-resident.
fn native_decoder_selected(decoder: &str) -> bool {
    let Ok(value) = std::env::var("ONNX_GENAI_PIPELINE_NATIVE_DECODER") else {
        return false;
    };
    value.split(',').map(str::trim).any(|entry| {
        !entry.is_empty()
            && (entry == decoder
                || matches!(
                    entry.to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on" | "all"
                ))
    })
}

/// Which pipeline components a flat autoregressive decode runs on the native
/// nxrt backend.
///
/// Unifies the two native selection sources so the pure-native backend and the
/// hybrid env-flag injection converge on the *same* component/decoder builders
/// (`build_step_component_session` / `build_native_pipeline_decoder`) — DRY, no
/// forked construction path:
/// - `EngineDecodeBackend::Native` selects **every** component natively.
/// - `EngineDecodeBackend::Ort` consults the per-component env flags
///   (`ONNX_GENAI_PIPELINE_NATIVE_DECODER` / `_NATIVE_STEP_COMPONENTS`), leaving
///   the default ORT decode path unchanged.
struct NativeComponentSelection {
    /// Drive the decoder through [`NativePipelineDecoder`] (device-resident KV).
    decoder: bool,
    /// every_step components to load as native `ComponentSession`s.
    step_components: BTreeSet<String>,
}

/// Build the backend-neutral [`ComponentSession`](onnx_genai_metadata::ComponentSession)
/// for one every_step component.
///
/// By default this borrows the already-loaded ORT session
/// ([`OrtComponentSessionRef`]) so the ORT decode path is behaviour-identical.
/// A component named in `native_components` is instead loaded and driven through
/// the native nxrt backend, so the same decode loop drives both backends through
/// the trait with no forked code path.
fn build_step_component_session<'a>(
    models: &'a PipelineModels,
    component: &str,
    native_components: &BTreeSet<String>,
    #[cfg_attr(not(feature = "native-backend"), allow(unused_variables))]
    native_device: &crate::native_decode_device::NativeDecodeDevice,
    #[cfg(feature = "cuda")] policy: onnx_runtime_ep_cuda::DeviceOffloadPolicy,
    #[cfg(feature = "cuda")] governor: std::sync::Arc<
        dyn onnx_runtime_memory_governor::MemoryGovernor + Send + Sync,
    >,
    #[cfg(feature = "cuda")] manager: onnx_runtime_memory_governor::ProcessMemoryManager,
) -> anyhow::Result<Box<dyn onnx_genai_metadata::ComponentSession + 'a>> {
    if native_components.contains(component) {
        #[cfg(feature = "native-backend")]
        {
            let path = models
                .directory
                .model_paths
                .get(component)
                .with_context(|| {
                    format!("native every_step component '{component}' has no model path")
                })?;
            // Run the native every_step component on the same device the native
            // decoder targets: a CUDA pipeline keeps the embedder on the GPU (so
            // an embeds-driven decoder's per-token embedding is a native CUDA EP
            // forward, mirroring the `muse_decode` raw-session setup), while a
            // CPU pipeline keeps it on CPU. Inputs/outputs still cross the host
            // `ComponentTensor` seam, so only one small `inputs_embeds` row
            // round-trips host<->device per step; the decoder's KV never does.
            // Under CUDA the governor and process manager are threaded into this
            // builder, so the component's provider is constructed already
            // governed; the loader still adopts the governor afterwards, which
            // is what creates the weight cache's mapped-byte allowance.
            // Without CUDA there is no governor in scope here at all: every_step
            // components are built before the resource governor, which is itself
            // sized from the models on disk. The lazily loaded prompt components
            // (routing.rs) are the ones that can and do adopt it.
            #[cfg(feature = "cuda")]
            let memory = crate::native_component::ComponentMemory::GovernedCuda {
                policy,
                governor,
                manager,
            };
            #[cfg(not(feature = "cuda"))]
            let memory = crate::native_component::ComponentMemory::SelfProvisioned(None);
            let native = crate::native_component::NativeComponentSession::load(
                path,
                native_device.clone(),
                memory,
            )
            .with_context(|| format!("failed to load native every_step component '{component}'"))?;
            return Ok(Box::new(native));
        }
        #[cfg(not(feature = "native-backend"))]
        {
            anyhow::bail!(
                "every_step component '{component}' was requested on the native backend via \
                 ONNX_GENAI_PIPELINE_NATIVE_STEP_COMPONENTS, but this build was compiled without \
                 the 'native-backend' feature. Rebuild with `--features native-backend`."
            );
        }
    }
    let session = models
        .session(component)
        .with_context(|| format!("pipeline every_step component '{component}' was not loaded"))?;
    Ok(Box::new(onnx_genai_ort::OrtComponentSessionRef::new(
        session,
    )))
}

/// Resolve the decoder's KV context ceiling for the native pipeline path.
///
/// The native decoder's per-directory model path (`.../decoder/model.onnx`) has
/// no metadata sidecar of its own, so the single-decoder auto-resolution from
/// the model directory cannot see the model's `max_sequence_length`. Without it
/// the CUDA KV capacity falls back to `usize::MAX` and the mask reservation
/// overflows before decode can start. Read the pipeline package's declared
/// context length instead: the native `inference_metadata.{yaml,yml,json}`
/// sidecar's `model.max_sequence_length` first, then the compatibility
/// `genai_config.json` context length. `None` preserves the prior unbounded
/// behavior for packages that declare neither.
#[cfg_attr(
    not(feature = "native-backend"),
    allow(
        dead_code,
        reason = "the declared context length is read only by the native decoder"
    )
)]
fn pipeline_metadata_max_len(directory: &onnx_genai_ort::PipelineModelDirectory) -> Option<usize> {
    if let Some(len) = directory
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.model.as_ref())
        .and_then(|model| model.max_sequence_length)
    {
        return Some(len);
    }

    onnx_genai_genai_config::find_in_dir(&directory.root)
        .and_then(|path| onnx_genai_genai_config::load(&path).ok())
        .and_then(|config| config.max_sequence_length())
}

fn pipeline_metadata_prefill_chunk_size(
    directory: &onnx_genai_ort::PipelineModelDirectory,
) -> Option<usize> {
    directory
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.model.as_ref())
        .and_then(|model| model.runtime_configurable.as_ref())
        .and_then(|runtime| runtime.chunked_prefill.as_ref())
        .and_then(|chunked| chunked.chunk_size)
        .filter(|size| *size > 0)
}

fn pipeline_metadata_generation_defaults(
    directory: &onnx_genai_ort::PipelineModelDirectory,
) -> Option<onnx_genai_metadata::GenerationDefaults> {
    directory
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.generation.clone())
}

fn pipeline_metadata_eos_token_ids(
    directory: &onnx_genai_ort::PipelineModelDirectory,
) -> Vec<TokenId> {
    directory
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.tokens.as_ref())
        .and_then(|tokens| tokens.eos_token_id.as_ref())
        .into_iter()
        .flatten()
        .filter_map(|id| TokenId::try_from(*id).ok())
        .collect()
}

/// Build the native device-KV [`PipelineDecoderComponent`] for `decoder`, loading
/// its ONNX model as a [`NativeDecodeSession`](crate::native_decode::NativeDecodeSession)
/// on the native backend so its KV cache stays session-resident across steps.
///
/// Returns an owned (`'static`) boxed decoder — unlike the ORT decoder it borrows
/// nothing from the pipeline decode state. Requesting the native decoder in a
/// build without the `native-backend` feature is a clear error, mirroring
/// [`build_step_component_session`].
#[cfg_attr(not(feature = "native-backend"), allow(unused_variables))]
fn build_native_pipeline_decoder(
    models: &PipelineModels,
    decoder: &str,
    config_device: Option<&crate::native_decode_device::NativeDecodeDevice>,
    memory_strategy_plan: &MemoryStrategyPlan,
    #[cfg(feature = "cuda")] governor: std::sync::Arc<
        dyn onnx_runtime_memory_governor::MemoryGovernor + Send + Sync,
    >,
    #[cfg(feature = "cuda")] manager: onnx_runtime_memory_governor::ProcessMemoryManager,
) -> anyhow::Result<Box<dyn PipelineDecoderComponent + 'static>> {
    #[cfg(feature = "native-backend")]
    {
        #[cfg(not(feature = "cuda"))]
        let _ = memory_strategy_plan;
        let path =
            models.directory.model_paths.get(decoder).with_context(|| {
                format!("native pipeline decoder '{decoder}' has no model path")
            })?;
        // Thread the pipeline-declared io spec so an inputs_embeds decoder binds
        // its sequence source / KV pairs from metadata rather than guessing.
        let io = models
            .directory
            .spec
            .models
            .get(decoder)
            .and_then(|component| component.io.as_ref());
        // Device precedence: ONNX_GENAI_PIPELINE_NATIVE_DECODER_DEVICE overrides
        // (back-compat / the deterministic parity fixture), else the engine's
        // configured `native_device` (so `--ep cuda` decodes on the GPU), else
        // CPU. A CUDA device keeps the KV cache resident on the CUDA EP (Inc3a);
        // one token's embedding uploads host->device per step, KV never
        // round-trips.
        let native = crate::pipeline::NativePipelineDecoder::load(
            path,
            native_decoder_device(config_device),
            io,
            pipeline_metadata_max_len(&models.directory),
            #[cfg(feature = "cuda")]
            crate::engine::cuda_policy_from_memory_strategy_plan(memory_strategy_plan),
            #[cfg(feature = "cuda")]
            governor,
            #[cfg(feature = "cuda")]
            manager,
        )?;
        // #1362: the pipeline's ORT decoders have always honored the declared
        // chunk size; the native decoder ignored it, so a prompt's prefill ran
        // as one forward and peak device memory scaled with prompt length.
        let mut native = native;
        native.set_prefill_chunk_size(pipeline_metadata_prefill_chunk_size(&models.directory));
        Ok(Box::new(native))
    }
    #[cfg(not(feature = "native-backend"))]
    {
        let _ = models;
        let _ = config_device;
        let _ = memory_strategy_plan;
        anyhow::bail!(
            "decoder '{decoder}' was requested on the native backend via \
             ONNX_GENAI_PIPELINE_NATIVE_DECODER, but this build was compiled without the \
             'native-backend' feature. Rebuild with `--features native-backend`."
        )
    }
}

/// Resolve the native pipeline decoder's device.
///
/// Precedence: the `ONNX_GENAI_PIPELINE_NATIVE_DECODER_DEVICE` env override wins
/// when set (`cpu`, `cuda`, or `cuda:<index>` / `cuda=<index>`; any unrecognized
/// value falls back to CPU so a stray setting never silently changes the
/// device). When the env var is unset, honor the engine-configured
/// `native_device` (so `--ep cuda` decodes on the GPU), else default to CPU.
#[cfg(feature = "native-backend")]
fn native_decoder_device(
    config_device: Option<&crate::native_decode_device::NativeDecodeDevice>,
) -> crate::native_decode_device::NativeDecodeDevice {
    let env = std::env::var("ONNX_GENAI_PIPELINE_NATIVE_DECODER_DEVICE").ok();
    resolve_native_decoder_device(env.as_deref(), config_device)
}

/// Pure device-precedence logic behind [`native_decoder_device`], split out so
/// the env-override-wins / config-fallback behavior is unit-testable without
/// mutating process-global environment state.
#[cfg(feature = "native-backend")]
fn resolve_native_decoder_device(
    env_value: Option<&str>,
    config_device: Option<&crate::native_decode_device::NativeDecodeDevice>,
) -> crate::native_decode_device::NativeDecodeDevice {
    use crate::native_decode_device::NativeDecodeDevice;
    match env_value {
        Some(value) => parse_native_decoder_device_value(value),
        None => config_device.cloned().unwrap_or(NativeDecodeDevice::Cpu),
    }
}

/// Parse a `ONNX_GENAI_PIPELINE_NATIVE_DECODER_DEVICE` value: `cuda`,
/// `cuda:<index>` / `cuda=<index>`, or anything else (→ CPU).
#[cfg(feature = "native-backend")]
fn parse_native_decoder_device_value(
    value: &str,
) -> crate::native_decode_device::NativeDecodeDevice {
    use crate::native_decode_device::NativeDecodeDevice;
    let value = value.trim().to_ascii_lowercase();
    match value.strip_prefix("cuda") {
        Some(rest) => {
            let index = rest.trim_start_matches([':', '=']).trim();
            let index = if index.is_empty() {
                None
            } else {
                index.parse::<u32>().ok()
            };
            NativeDecodeDevice::Cuda { index }
        }
        None => NativeDecodeDevice::Cpu,
    }
}

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

    pub fn from_pipeline_dir_with_memory_authority_provider(
        pipeline_dir: &Path,
        config: EngineConfig,
        provider: Arc<dyn MemoryAuthorityProvider>,
    ) -> anyhow::Result<PipelineEngine> {
        PipelineEngine::from_dir_with_memory_authority_provider(pipeline_dir, config, provider)
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

    /// Resolved decoder execution backend.
    pub fn decode_backend(&self) -> EngineDecodeBackend {
        self.decode_backend
    }

    /// Model-authored sampling defaults declared by the pipeline package.
    pub fn generation_defaults(&self) -> Option<&onnx_genai_metadata::GenerationDefaults> {
        self.generation_defaults.as_ref()
    }

    /// Effective context limit for a request, combining the package metadata
    /// with an explicit per-request override.
    pub fn effective_max_context(&self, options: &GenerateOptions) -> Option<usize> {
        options.max_context.or(self.max_sequence_length)
    }

    /// Execution-provider placement reported by the loaded component sessions.
    pub fn execution_provider_status(&self) -> String {
        let mut summaries = self
            .models
            .sessions
            .values()
            .map(|session| session.execution_provider_status().summary())
            .collect::<Vec<_>>();
        summaries.sort();
        summaries.dedup();
        if summaries.is_empty() {
            "native".to_string()
        } else {
            summaries.join("; ")
        }
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
            None,
        )
    }

    /// Load a pipeline using a caller-owned device authority provider.
    pub fn from_dir_with_memory_authority_provider(
        pipeline_dir: &Path,
        config: EngineConfig,
        provider: Arc<dyn MemoryAuthorityProvider>,
    ) -> anyhow::Result<Self> {
        Self::build(
            pipeline_dir,
            config,
            &SchedulerRegistry::builtin(),
            SessionOptions::default(),
            Some(provider),
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
        Self::build(
            pipeline_dir,
            config,
            schedulers,
            SessionOptions::default(),
            None,
        )
    }

    fn build(
        pipeline_dir: &Path,
        config: EngineConfig,
        schedulers: &SchedulerRegistry,
        session_options: SessionOptions,
        authority_provider: Option<SharedMemoryAuthorityProvider>,
    ) -> anyhow::Result<Self> {
        // Explicit backend requests must fail before touching the model
        // directory. In particular, a binary without native support should
        // report the actionable rebuild error even when the path is invalid.
        let decode_backend = validate_pipeline_backend_request(config.decode_backend)?;
        let authority_domain = crate::engine::session_device_domain(&session_options)?;
        crate::engine::validate_shared_authority_limit(
            authority_provider.as_ref(),
            &authority_domain,
            config.limits.vram_limit,
        )?;
        let directory = PipelineModelDirectory::load(pipeline_dir)
            .map_err(|e| anyhow::anyhow!("Failed to resolve pipeline models: {e}"))?;
        let prefill_chunk_size = pipeline_metadata_prefill_chunk_size(&directory);
        let generation_defaults = pipeline_metadata_generation_defaults(&directory);
        let max_sequence_length = pipeline_metadata_max_len(&directory);
        let eos_token_ids = pipeline_metadata_eos_token_ids(&directory);
        let model_weights_bytes =
            directory
                .model_paths
                .values()
                .try_fold(0_u64, |total, path| {
                    total
                        .checked_add(onnx_genai_ort::model_weight_bytes(path))
                        .ok_or_else(|| anyhow::anyhow!("pipeline component weight size overflow"))
                })?;
        // Select ONE backend for the whole pipeline (never a mix). Explicit
        // backends resolve without touching the model directory (so a bad
        // request fails fast); `Auto` inspects the components' declared
        // operators, selecting native only when some component requires it.
        let backend = match decode_backend {
            EngineDecodeBackend::Ort => PipelineBackend::Ort,
            EngineDecodeBackend::Native => PipelineBackend::Native,
            EngineDecodeBackend::Auto => resolve_auto_pipeline_backend(&directory)?,
        };
        let plan = PipelinePlan::from_spec(&directory.spec, schedulers)?;
        // Resolve the native device from the session's execution providers, the
        // same way `engine::load` does for the standalone decoder.
        //
        // `config.native_device` is only ever `Some` when an operator named a
        // device explicitly, so reading it alone made `--ep cuda` resolve to
        // `Cpu` for every native pipeline session. The decoder survived that on
        // its own load path, which does consult the providers; the lazily loaded
        // prompt components (a vision tower) did not, and ran the whole encoder
        // on CPU with no warning -- a >20 minute prefill where the GPU took 13s.
        #[cfg(feature = "native-backend")]
        let resolved_native_device = match backend {
            PipelineBackend::Native => Some(crate::engine::resolve_native_decode_device(
                config.native_device.clone(),
                &session_options,
            )?),
            PipelineBackend::Ort => config.native_device.clone(),
        };
        let graph_memory = combine_graph_memory(
            directory
                .model_paths
                .values()
                .map(|path| analyze_model_memory(path)),
            matches!(&plan, PipelinePlan::Iterative(_)),
        );
        let minimum_useful_weight_budget_bytes = graph_memory
            .per_layer_weight_bytes
            .iter()
            .map(|layer| layer.bytes)
            .max()
            .unwrap_or(0);
        let memory_strategy_kv_config = match &plan {
            PipelinePlan::Autoregressive(ar) => {
                let path = directory.model_paths.get(&ar.decoder).with_context(|| {
                    format!("pipeline decoder '{}' has no model path", ar.decoder)
                })?;
                let decoder_io = directory
                    .spec
                    .models
                    .get(&ar.decoder)
                    .and_then(|component| component.io.as_ref());
                let kv_inputs = decoder_io
                    .and_then(|io| io.kv_inputs.clone())
                    .unwrap_or_default();
                let kv_outputs = decoder_io
                    .and_then(|io| io.kv_outputs.clone())
                    .unwrap_or_default();
                let graph_io = onnx_genai_ort::graph_io_from_model_path_for_kv_pairs(
                    path,
                    &kv_inputs,
                    &kv_outputs,
                )
                .with_context(|| {
                    format!(
                        "failed to read pipeline decoder '{}' KV graph I/O",
                        ar.decoder
                    )
                })?;
                let kv_model = infer_kv_model_info(
                    &graph_io,
                    decoder_io,
                    config.page_size,
                    config.kv_cache_dtype,
                )?;
                match kv_model.as_ref() {
                    Some(kv_model) => crate::engine::governor_kv_config(Some(kv_model), &config)?,
                    None => crate::engine::governor_no_paged_kv_config(&config)?,
                }
            }
            _ => crate::engine::governor_no_paged_kv_config(&config)?,
        };
        // Resolve the fractional VRAM limit against the real device capacity
        // when the native CUDA pipeline path is active; otherwise the
        // provisional 8 GiB constant would cap leases far below a large model's
        // resident weights on any GPU.
        #[cfg(all(feature = "cuda", feature = "native-backend"))]
        let pipeline_cuda_index = if backend == PipelineBackend::Native {
            native_decoder_device(resolved_native_device.as_ref()).cuda_index()
        } else {
            None
        };
        #[cfg(not(all(feature = "cuda", feature = "native-backend")))]
        let pipeline_cuda_index: Option<u32> = None;
        // The device (VRAM) capacity stays honestly `None` when it cannot be
        // measured (#947): it is reported verbatim as `resolved_device_budget`
        // and never borrows the host tier. The residency verdict is a separate,
        // still-knowable fact, sized against the physical hot tier the weights
        // will really occupy: the measured VRAM budget when the pipeline targets
        // a queryable device, else the measured host-RAM ceiling. Unknown device
        // *capacity* must not turn a resident model into `Unknown`.
        let resolved_vram_bytes = resolve_vram_limit_bytes(&config.limits, pipeline_cuda_index)?;
        let residency_ceiling_bytes =
            resolve_memory_strategy_hot_tier_bytes(&config.limits, pipeline_cuda_index)?;
        #[cfg(feature = "cuda")]
        let memory_strategy_overrides = crate::engine::memory_strategy_overrides_from_cuda_env(
            onnx_runtime_ep_cuda::DeviceOffloadPolicy::from_env(),
        );
        #[cfg(not(feature = "cuda"))]
        let memory_strategy_overrides = crate::engine::MemoryStrategyOverrides::default();
        #[cfg(all(feature = "cuda", feature = "native-backend"))]
        let native_cuda_plan = backend == PipelineBackend::Native
            && matches!(
                native_decoder_device(resolved_native_device.as_ref()),
                crate::native_decode_device::NativeDecodeDevice::Cuda { .. }
            );
        #[cfg(not(all(feature = "cuda", feature = "native-backend")))]
        let native_cuda_plan = false;
        // #971: on the native CPU pipeline path each component's MatMulNBits
        // kernel may build a resident dequantised f32 weight cache held for the
        // session. Ask the CPU EP (which owns kernel dispatch) how many extra
        // bytes each component's cache costs, summed, so the plan accounts for
        // the real resident footprint instead of the on-disk size (#947). CUDA
        // and ORT use different kernels, so it never applies there.
        #[cfg(feature = "native-backend")]
        let pipeline_resident_f32_cache_bytes = if backend == PipelineBackend::Native
            && !native_cuda_plan
        {
            directory
                .model_paths
                .values()
                .map(|path| {
                    onnx_runtime_loader::load_model(path)
                        .map(|graph| onnx_runtime_ep_cpu::resident_dequant_f32_cache_bytes(&graph))
                        .unwrap_or(0)
                })
                .fold(0_u64, |total, bytes| total.saturating_add(bytes))
        } else {
            0
        };
        #[cfg(not(feature = "native-backend"))]
        let pipeline_resident_f32_cache_bytes = 0_u64;
        // #755: managed no-spill VMM is the default on the native CUDA pipeline
        // path unless the legacy allocator opt-out is set. Other backends keep
        // the pre-#755 explicit-byte-limit trigger.
        #[cfg(all(feature = "cuda", feature = "native-backend"))]
        let pipeline_managed_vmm = if native_cuda_plan {
            crate::engine::managed_vmm_default_enabled()
        } else {
            matches!(config.limits.vram_limit, crate::ResourceLimit::Bytes(_))
        };
        #[cfg(not(all(feature = "cuda", feature = "native-backend")))]
        let pipeline_managed_vmm =
            matches!(config.limits.vram_limit, crate::ResourceLimit::Bytes(_));
        let memory_strategy_plan = build_memory_strategy_plan(MemoryStrategyPlanInput {
            config: &config,
            resolved_vram_bytes,
            residency_ceiling_bytes,
            model_weight_bytes: model_weights_bytes,
            resident_f32_cache_bytes: pipeline_resident_f32_cache_bytes,
            kv_config: memory_strategy_kv_config,
            graph: graph_memory,
            required_device_non_weight_bytes: 0,
            minimum_useful_weight_budget_bytes,
            #[cfg(feature = "cuda")]
            default_dynamic_device_budget_bytes: Some(
                onnx_runtime_ep_cuda::DEFAULT_DEVICE_OFFLOAD_BUDGET_BYTES,
            ),
            #[cfg(not(feature = "cuda"))]
            default_dynamic_device_budget_bytes: None,
            inferred_policy_enabled: pipeline_managed_vmm
                || matches!(config.limits.vram_limit, crate::ResourceLimit::Bytes(_)),
            managed_vmm: pipeline_managed_vmm,
            overrides: memory_strategy_overrides,
            advisory_only: !native_cuda_plan,
            // #864: WDDM shared-memory fallback is a Windows platform property.
            shared_memory_weight_fallback: cfg!(windows),
            force_managed_weight_streaming: crate::engine::force_managed_weight_streaming_enabled(),
        });
        log_memory_strategy_plan(&memory_strategy_plan, "pipeline");
        // #971: tell the CPU EP whether the governor admitted the resident f32
        // decode cache for the native pipeline path (no-op for ORT/CUDA, which
        // pass 0 bytes and thus always report admitted).
        #[cfg(feature = "native-backend")]
        onnx_runtime_ep_cpu::set_resident_dequant_f32_cache_enabled(
            memory_strategy_plan.f32_weight_cache_admitted,
        );
        // #1027: same admission verdict governs the int4 accuracy_level=0 MLAS
        // SQNBit packed buffer (folded into `resident_f32_cache_bytes`); when
        // declined the kernel keeps the borrowed zero-copy int4 path.
        #[cfg(feature = "native-backend")]
        onnx_runtime_ep_cpu::set_mlas_sqnbit_packing_enabled(
            memory_strategy_plan.f32_weight_cache_admitted,
        );
        #[cfg(all(feature = "cuda", feature = "native-backend"))]
        let authority_domain = if backend == PipelineBackend::Native {
            match native_decoder_device(resolved_native_device.as_ref()) {
                crate::native_decode_device::NativeDecodeDevice::Cuda { index } => {
                    crate::memory_authority::DeviceCompatibilityDomain::Cuda(index.unwrap_or(0))
                }
                _ => authority_domain,
            }
        } else {
            authority_domain
        };
        // Reserve every component's package bytes before constructing the
        // first session. Native CUDA components and their VMM pool share this
        // same authority, including in standalone pipelines.
        let component_weight_reservation_bytes = if native_cuda_plan && pipeline_managed_vmm {
            // Managed VMM charges physical handles to this authority as
            // components commit them. Reserving package bytes here would
            // charge the same weights twice.
            0
        } else {
            model_weights_bytes
        };
        let resource_governor = component_governor(
            &config,
            None,
            model_weights_bytes,
            component_weight_reservation_bytes,
            pipeline_cuda_index,
            authority_provider.as_ref(),
            &authority_domain,
        )?;
        #[cfg(all(feature = "cuda", feature = "native-backend"))]
        let native_cuda_authority = if backend == PipelineBackend::Native {
            match native_decoder_device(resolved_native_device.as_ref()) {
                crate::native_decode_device::NativeDecodeDevice::Cuda { .. } => {
                    Some(resource_governor.device_authority())
                }
                _ => None,
            }
        } else {
            None
        };
        #[cfg(all(feature = "cuda", not(feature = "native-backend")))]
        let native_cuda_authority = None;
        if backend == PipelineBackend::Native {
            // The native backend constructs every declared component through the
            // backend-neutral `ComponentSession` seam (GAP 1). When the crate is
            // built without the `native-backend` feature there is nothing to
            // construct, so fail fast with an actionable rebuild error.
            #[cfg(not(feature = "native-backend"))]
            {
                return Err(native_backend_not_compiled_error());
            }
            // With the feature present the pipeline is constructed below exactly
            // like the ORT path; the flat autoregressive decode loop then drives
            // every component natively (native `ComponentSession`s + a
            // `NativePipelineDecoder` keeping KV device-resident) via
            // `native_component_selection` (GAP-3 Inc-A). Plans other than flat
            // autoregressive are not yet wired for native and are rejected once
            // the plan is known (below), rather than at construction.
        }
        let native_ort_skips = if backend == PipelineBackend::Native {
            match &plan {
                // The native backend runs every component through native nxrt
                // sessions: the decoder and every_step components in the decode
                // loop, and any prompt-phase component (e.g. a vision encoder or
                // an embeds fuser) lazily in the prologue. Building ORT sessions
                // for them would be redundant and, for graphs using native-only
                // operators (bf16 `Where`, int4 `MatMulNBits`), would make ORT
                // reject the model at load. So skip ORT for all of them and keep
                // only their session-free graph I/O for decode resolution.
                PipelinePlan::Autoregressive(_) => directory
                    .model_paths
                    .keys()
                    .cloned()
                    .collect::<BTreeSet<_>>(),
                _ => BTreeSet::new(),
            }
        } else {
            BTreeSet::new()
        };
        let models = if backend == PipelineBackend::Native {
            // The native flat-AR backend drives every component through native
            // nxrt sessions, so ORT sessions would be redundant and can reject
            // native-only operators. Prompt-phase components run natively in the
            // prologue (built lazily by `run_prompt_phase_components`).
            PipelineModels::load_with_ort_session_filter(pipeline_dir, session_options, |name| {
                !native_ort_skips.contains(name)
            })
        } else {
            PipelineModels::load_with_options(pipeline_dir, session_options)
        }
        .map_err(|e| anyhow::anyhow!("Failed to load pipeline models: {e}"))?;
        // GAP-3 Inc-A wires native decode only for the flat autoregressive plan.
        // Any other strategy on the native backend is surfaced with a precise,
        // actionable error naming the still-unwired path (rather than a blanket
        // rejection or a silent mis-route through the ORT-shaped loop).
        #[cfg(feature = "native-backend")]
        if backend == PipelineBackend::Native && !matches!(plan, PipelinePlan::Autoregressive(_)) {
            return Err(native_pipeline_plan_unsupported(&models.directory, &config));
        }
        let memoizable_components = deterministic_components(&models.directory);
        // Only autoregressive pipelines drive a token-by-token decode loop and
        // therefore need a `DecodeState` + KV model info. Single-pass and
        // iterative (diffusion) pipelines run tensors through `run_pipeline`.
        let mut paged: Option<PipelinePagedKv> = None;
        let mut kv_governor = None;
        let (decoder_state, tokenizer_component, fixed_state_budget_bytes) = match &plan {
            PipelinePlan::Autoregressive(ar) => {
                let decoder = models
                    .graph_io(&ar.decoder)
                    .with_context(|| format!("pipeline decoder '{}' was not loaded", ar.decoder))?;
                let decoder_io = models
                    .directory
                    .spec
                    .models
                    .get(&ar.decoder)
                    .and_then(|component| component.io.as_ref());
                let kv_model = infer_kv_model_info(
                    decoder,
                    decoder_io,
                    config.page_size,
                    config.kv_cache_dtype,
                )?;
                let kv_config = match kv_model.as_ref() {
                    Some(kv_model) => crate::engine::governor_kv_config(Some(kv_model), &config)?,
                    None => crate::engine::governor_no_paged_kv_config(&config)?,
                };
                let component_governor = EngineResourceGovernor::new_for_shared_pipeline_kv(
                    config.limits.clone(),
                    config.allow_runtime_override,
                    kv_config,
                    resource_governor.snapshot().vram.used,
                    pipeline_cuda_index,
                    authority_provider.as_ref(),
                    &authority_domain,
                )
                .context("failed to resolve the shared pipeline KV memory budget")?;
                let fixed_state_budget_bytes =
                    component_governor.snapshot().resolved_limits.host_ram_bytes;
                let pipeline_pages = match kv_model.as_ref() {
                    Some(kv_model) => crate::engine::kv_pages_for_budget(
                        component_governor.snapshot().derived_budget.kv_bytes,
                        component_governor.snapshot().resolved_limits.host_ram_bytes,
                        config.scheduler.max_total_tokens,
                        kv_model.tensor_config.page_size,
                        kv_model.tensor_config.dtype,
                        &kv_model.layer_configs,
                    ),
                    None => 0,
                };
                // A zero page size makes `div_ceil` panic and the page-boundary
                // walk below produce zeros forever, so it is refused rather than
                // carried into arithmetic that assumes it is positive.
                paged = match kv_model.filter(|kv_model| kv_model.tensor_config.page_size > 0) {
                    Some(kv_model) => Some(PipelinePagedKv {
                        cache: component_governor
                            .plan()
                            .kv_pool(
                                crate::engine::memory_plan::Holder::PipelineKvPool,
                                kv_model.tensor_config.page_size,
                                kv_model.tensor_config.dtype,
                                kv_model.layer_configs.clone(),
                                pipeline_pages,
                            )
                            .with_context(|| {
                                format!(
                                    "cannot allocate the pipeline KV page pool: {pipeline_pages} \
                                 page(s) across {} layer(s) do not fit the resolved KV/host \
                                 memory budget",
                                    kv_model.layer_configs.len()
                                )
                            })?,
                        kv_model,
                        prefix: PrefixCache::new(),
                        active: None,
                    }),
                    None => None,
                };
                kv_governor = Some(component_governor);
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
            resource_governor,
            _kv_governor: kv_governor,
            memory_strategy_plan,
            plan,
            decode_backend: match backend {
                PipelineBackend::Ort => EngineDecodeBackend::Ort,
                PipelineBackend::Native => EngineDecodeBackend::Native,
            },
            generation_defaults,
            max_sequence_length,
            eos_token_ids,
            decoder_state,
            tokenizer_component,
            fixed_state_budget_bytes,
            component_cache: RefCell::new(ComponentOutputCache::new(
                usize::try_from(config.pipeline_cache_bytes).unwrap_or(usize::MAX),
            )),
            memoizable_components,
            retained: None,
            native_retained_decoder: None,
            prefill_chunk_size,
            paged,
            #[cfg(feature = "native-backend")]
            native_device: resolved_native_device,
            #[cfg(not(feature = "native-backend"))]
            native_device: config.native_device.clone(),
            native_prompt_sessions: RefCell::new(BTreeMap::new()),
            #[cfg(feature = "cuda")]
            native_cuda_authority,
        })
    }

    pub fn resource_snapshot(&self) -> onnx_genai_scheduler::GovernorSnapshot {
        #[cfg(feature = "cuda")]
        {
            let mut snapshot = self.resource_governor.snapshot();
            if let Some(authority) = &self.native_cuda_authority {
                snapshot.vram.used = authority.used_bytes();
                snapshot.vram.limit = authority.limit_bytes();
                snapshot.vram.headroom = authority.headroom_bytes();
                snapshot.resolved_limits.vram_bytes = Some(authority.limit_bytes());
            }
            snapshot
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.resource_governor.snapshot()
        }
    }

    pub fn memory_strategy_plan(&self) -> &MemoryStrategyPlan {
        &self.memory_strategy_plan
    }

    pub fn models(&self) -> &PipelineModels {
        &self.models
    }

    pub fn device_authority(&self) -> crate::memory_authority::DeviceMemoryAuthority {
        #[cfg(feature = "cuda")]
        if let Some(authority) = &self.native_cuda_authority {
            return authority.clone();
        }
        self.resource_governor.device_authority()
    }

    pub fn set_vram_limit(
        &self,
        limit: onnx_genai_scheduler::ResourceLimit,
    ) -> Result<onnx_genai_scheduler::GovernorReconfigureOutcome, crate::engine::EngineGovernorError>
    {
        self.resource_governor.set_vram_limit(limit)
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
        self.generate_with_callbacks(pipeline_request, None, callback)
    }

    /// Generate text with a pre-token admission callback. For a native routed
    /// decoder, admission fires only after exact step inputs have prepared and
    /// reserved governed workspace.
    pub fn generate_with_callbacks(
        &mut self,
        pipeline_request: PipelineGenerateRequest,
        mut admission: Option<&mut dyn FnMut()>,
        callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        // A nested-AR (multi-decoder TTS) pipeline drives its own outer/inner
        // loops; `generate` returns the flattened per-frame code tokens (use
        // `synthesize` to also run the post-decode vocoder into a waveform).
        if matches!(self.plan, PipelinePlan::NestedAutoregressive(_)) {
            if let Some(admitted) = admission.as_mut() {
                admitted();
            }
            return self
                .run_nested_autoregressive(pipeline_request)
                .map(|(result, _pool)| result);
        }
        self.run_autoregressive(pipeline_request, admission, callback)
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
        let (generation, mut tensors) = self.run_autoregressive(pipeline_request, None, None)?;
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

    /// Add noise to an encoded diffusion latent at the scheduler state for `step`.
    pub fn diffusion_add_noise(
        &self,
        step: usize,
        num_steps: usize,
        original: &Value,
        noise: &Value,
    ) -> anyhow::Result<Value> {
        let PipelinePlan::Iterative(iterative) = &self.plan else {
            anyhow::bail!("pipeline is not iterative");
        };
        if step > num_steps {
            anyhow::bail!("start step ({step}) must be <= num_steps ({num_steps})");
        }
        let rebuilt;
        let scheduler = if num_steps == iterative.num_steps {
            iterative.scheduler.as_ref()
        } else {
            rebuilt = iterative
                .scheduler_spec
                .as_ref()
                .map(|spec| iterative.scheduler_registry.build(spec, num_steps))
                .transpose()?;
            rebuilt.as_ref()
        }
        .context("iterative pipeline declares no diffusion scheduler")?;
        scheduler.add_noise(step, num_steps, original, noise)
    }

    /// Generate an image and retain the post-scheduler latent for every step.
    pub fn generate_image(&mut self, request: ImageRequest) -> anyhow::Result<ImageStream> {
        let mut steps = Vec::new();
        let output = self.generate_image_with_callback(request, &mut |step| {
            let latents = step
                .latents
                .iter()
                .map(|(endpoint, value)| Ok((endpoint.clone(), clone_value(value)?)))
                .collect::<anyhow::Result<_>>()?;
            steps.push(ImageStep {
                step: step.step,
                latents,
            });
            Ok(())
        })?;
        Ok(ImageStream { steps, output })
    }

    /// Generate an image while observing each post-scheduler latent immediately.
    ///
    /// The callback is invoked once per denoise step for every scheduler through
    /// the shared `step` / `step_with_noise` dispatch.
    pub fn generate_image_with_callback(
        &mut self,
        request: ImageRequest,
        callback: &mut ImageStepCallback<'_>,
    ) -> anyhow::Result<ImageOutput> {
        let ImageRequest {
            pipeline,
            image_output,
        } = request;
        let tensors = self.run_iterative_with_callback(pipeline, Some(callback))?;
        let image_endpoint = self.image_output_endpoint(image_output.as_deref())?;
        let image = tensors.get(&image_endpoint).with_context(|| {
            format!("iterative pipeline did not produce requested image '{image_endpoint}'")
        })?;
        let image = clone_value(image)?;
        let latents = self.final_image_latents(&tensors)?;
        Ok(ImageOutput { image, latents })
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

    fn image_output_endpoint(&self, requested: Option<&str>) -> anyhow::Result<String> {
        if let Some(endpoint) = requested {
            return Ok(endpoint.to_string());
        }
        let PipelinePlan::Iterative(plan) = &self.plan else {
            anyhow::bail!("generate_image() requires an iterative pipeline");
        };
        let component = plan.final_components.last().with_context(|| {
            "generate_image() requires a final image component or ImageRequest::with_image_output()"
        })?;
        let session = self
            .models
            .graph_io(component)
            .with_context(|| format!("final image component '{component}' was not loaded"))?;
        let outputs = session.output_names();
        if outputs.len() != 1 {
            anyhow::bail!(
                "final image component '{component}' has {} outputs; select one with \
                 ImageRequest::with_image_output()",
                outputs.len()
            );
        }
        Ok(format!("{component}.{}", outputs[0]))
    }

    fn final_image_latents(&self, tensors: &PipelineTensors) -> anyhow::Result<PipelineTensors> {
        let PipelinePlan::Iterative(plan) = &self.plan else {
            anyhow::bail!("generate_image() requires an iterative pipeline");
        };
        plan.loop_edges
            .iter()
            .map(|(_, input)| {
                let endpoint = format!("{}.{}", plan.denoiser, input);
                let value = tensors.get(&endpoint).with_context(|| {
                    format!("final iterative latent '{endpoint}' was not produced")
                })?;
                Ok((endpoint, clone_value(value)?))
            })
            .collect()
    }

    fn tokenizer(&self) -> anyhow::Result<&Tokenizer> {
        self.models
            .tokenizer_for(&self.tokenizer_component)
            .with_context(|| format!("no tokenizer available for '{}'", self.tokenizer_component))
    }

    /// Drop every reusable prefix this pipeline is holding, so the next
    /// generation recomputes its prompt from scratch. Returns how many KV pages
    /// were freed from the paged prefix cache, if any.
    ///
    /// Two things are cleared, because a pipeline reuses prefixes through
    /// whichever is active: the paged [`PrefixCache`], and the `retained`
    /// context that carries a native decoder's device-resident KV from one
    /// sequential request to the next.
    ///
    /// Benchmarks need this. A harness that replays one prompt — which is what
    /// warmup runs do — otherwise answers each measured generation almost
    /// entirely out of a retained prefix, and reports a "prefill" that processed
    /// a single token and does not vary with prompt length (#1529).
    pub fn clear_prefix_cache(&mut self) -> usize {
        self.retained = None;
        let Some(paged) = self.paged.as_mut() else {
            return 0;
        };
        paged
            .prefix
            .evict_lru(usize::MAX, &mut paged.cache.page_table)
            .len()
    }

    /// Encode text with the same tokenizer this pipeline uses for prompts.
    ///
    /// The public seam benchmarks need to report how many prompt tokens a
    /// generation actually processed, and to build prompts of an exact token
    /// length. Without it a harness has to re-open `tokenizer.json` itself and
    /// hope it picked the same component this pipeline routes prompts through.
    pub fn tokenize(&self, text: &str) -> anyhow::Result<Vec<TokenId>> {
        self.tokenizer()?.encode(text).map_err(|e| {
            anyhow::anyhow!(
                "failed to tokenize input text with the pipeline's tokenizer: {e}; \
                 verify the model directory contains a valid tokenizer.json"
            )
        })
    }

    /// Decode token ids back to text with this pipeline's tokenizer, the
    /// inverse seam of [`Pipeline::tokenize`].
    pub fn detokenize(&self, tokens: &[TokenId]) -> anyhow::Result<String> {
        self.tokenizer()?
            .decode(tokens)
            .map_err(|e| anyhow::anyhow!("failed to detokenize token ids: {e}"))
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

    /// TEST-SUPPORT (native-backend only): read back the paged KV bytes that
    /// were published for `request`'s prompt prefix, non-destructively.
    ///
    /// Reconstructs the exact prefix key the paged decode path publishes under
    /// (`digest_request_identity` + `prefix_key`), looks it up in the prefix
    /// cache *without* touching page refcounts, attaches the matched pages to a
    /// throwaway sequence purely to read them, and materializes the per-layer
    /// K/V into contiguous buffers. The throwaway sequence is then dropped
    /// without freeing the pages — they belong to the prefix cache, and the
    /// attach did not retain them — so the cache is left exactly as found.
    ///
    /// This exists because the Inc-C paged-reuse parity test cannot see KV
    /// geometry through token equality alone: the `tiny-gemma4-vlm` fixture's
    /// argmax is invariant to the reused-prefix KV, so a key/value swap or a
    /// fully-zeroed mirror still yields identical tokens. Comparing these
    /// materialized bytes between the native-mirrored and ORT-mirrored caches
    /// gives the test discriminating power over the mirror geometry.
    ///
    /// Returns `None` when nothing is published for the prefix (no digestable
    /// inputs, no paged cache, or no matched pages).
    #[cfg(feature = "native-backend")]
    pub fn materialize_published_prefix_kv(
        &mut self,
        request: &PipelineGenerateRequest,
    ) -> anyhow::Result<Option<onnx_genai_kv::MaterializedKv>> {
        let Some(inputs) = Self::digest_request_identity(request) else {
            return Ok(None);
        };
        let prompt_tokens = tokenize_with(self.tokenizer()?, &request.request.prompt)?;
        let Some(paged) = self.paged.as_mut() else {
            return Ok(None);
        };
        let key = prefix_key(inputs, &prompt_tokens);
        let (matched_tokens, page_ids) = paged.prefix.lookup(&key);
        let reusable = matched_tokens.saturating_sub(PREFIX_KEY_PREAMBLE);
        if reusable == 0 || page_ids.is_empty() {
            return Ok(None);
        }
        let pages_needed = reusable.div_ceil(paged.cache.page_table.page_size);
        let pages = page_ids.into_iter().take(pages_needed).collect::<Vec<_>>();
        let seq = paged.cache.create_sequence();
        let materialized = (|| {
            attach_pages_to_sequence(&mut paged.cache, seq, &pages, reusable)?;
            paged
                .cache
                .materialize_sequence(seq)
                .map_err(anyhow::Error::from)
        })();
        // Forget the throwaway sequence's borrowed page list without freeing the
        // pages: `attach_pages_to_sequence` did not retain them, so the prefix
        // cache remains their sole owner and is left untouched.
        paged.cache.page_table.remove_sequence(seq);
        materialized.map(Some)
    }

    /// Counters describing what the pipeline's reuse caches did.
    pub fn cache_stats(&self) -> PipelineCacheStats {
        self.component_cache.borrow().stats()
    }

    /// Clear the per-generation counters reported by [`cache_stats`](Self::cache_stats).
    pub fn reset_cache_stats(&self) {
        self.component_cache.borrow_mut().reset_stats();
    }
}

fn tokenize_with(tokenizer: &Tokenizer, prompt: &GeneratePrompt) -> anyhow::Result<Vec<TokenId>> {
    match prompt {
        GeneratePrompt::TokenIds(tokens) => Ok(tokens.clone()),
        GeneratePrompt::Text(text) => tokenizer
            .encode(text)
            .map_err(|e| anyhow::anyhow!("Failed to tokenize prompt: {e}")),
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
pub(crate) struct StepComponentBinding {
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
pub(crate) struct StepComponentInput {
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
    /// Inner decoder OUTPUT port carrying the per-code embedding threaded into
    /// the next inner step's `inputs_embeds` seed. Declared explicitly via
    /// `pipeline.strategy.inner_embedding_output`; never inferred by tensor name.
    inner_embedding_output: String,
    /// Outer decoder logits output port, from the outer component's explicit
    /// `io.logits_output`. Argmax reads exactly this port; never name-guessed.
    outer_logits_output: String,
    /// Inner decoder logits output port, from the inner component's explicit
    /// `io.logits_output`. Argmax reads exactly this port; never name-guessed.
    inner_logits_output: String,
    /// Outer decoder explicit I/O metadata, threaded into decode-state resolution
    /// so token/position/KV ports come from declared metadata rather than a
    /// tensor-shape guess. `None` only when the component declares no `io` block.
    outer_io: Option<ModelIoSpec>,
    /// Inner decoder explicit I/O metadata, threaded into decode-state resolution
    /// for the same reason as `outer_io`.
    inner_io: Option<ModelIoSpec>,
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

/// Convert a pool ORT [`Value`] into a backend-neutral [`ComponentTensor`] for
/// the [`ComponentSession`](onnx_genai_metadata::ComponentSession) seam.
///
/// This is the pipeline side of the value-type seam: the decode-loop pool holds
/// ORT `Value`s, but every_step components run through the backend-neutral trait
/// whose boundary is a host-resident `ComponentTensor` (raw little-endian element
/// bytes). The copy is `numel * dtype.size_of()` bytes; for the small every_step
/// embedding outputs this is negligible relative to the decoder step.
fn value_to_component_tensor(
    value: &Value,
) -> anyhow::Result<onnx_genai_metadata::ComponentTensor> {
    let dtype = onnx_genai_metadata::ComponentDataType::from(value.dtype());
    let bytes = value.to_raw_bytes()?;
    onnx_genai_metadata::ComponentTensor::from_raw(dtype, value.shape().to_vec(), bytes)
        .map_err(Into::into)
}

/// Convert a [`ComponentTensor`] produced by a component back into a pool ORT
/// [`Value`]. Inverse of [`value_to_component_tensor`].
fn component_tensor_to_value(
    tensor: &onnx_genai_metadata::ComponentTensor,
) -> anyhow::Result<Value> {
    let dtype = DataType::from(tensor.dtype());
    Value::from_raw_bytes(tensor.as_bytes().to_vec(), tensor.shape(), dtype).map_err(Into::into)
}

/// Build the running-token seed as a neutral `int64` [`ComponentTensor`] of shape
/// `[1, seq]`, matching the ORT `Value::from_slice_i64` the loop previously fed.
fn token_seed_component_tensor(
    ids: &[i64],
) -> anyhow::Result<onnx_genai_metadata::ComponentTensor> {
    let bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    onnx_genai_metadata::ComponentTensor::from_raw(
        onnx_genai_metadata::ComponentDataType::Int64,
        vec![1, ids.len() as i64],
        bytes,
    )
    .map_err(Into::into)
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

        // The inner decoder threads its own per-code embedding on later steps.
        // The exact output port is declared explicitly (it is shape-indistinguish-
        // able from other float outputs); a missing declaration is an actionable
        // error rather than a tensor-name guess.
        let inner_embedding_output = nested.inner_embedding_output.clone().context(
            "nested_autoregressive strategy is missing \
                 'inner_embedding_output' (the inner decoder output port threaded \
                 across inner steps)",
        )?;
        if inner_embedding_output.is_empty() {
            anyhow::bail!("nested_autoregressive 'inner_embedding_output' must not be empty");
        }

        // Both decoders' logits ports are read by argmax each step. A logits
        // output is shape-indistinguishable from other float outputs, so each
        // port is taken from the component's explicit `io.logits_output`; a
        // missing declaration is an actionable error, never a name guess.
        let outer_logits_output = require_component_logits_output(spec, &outer)?;
        let inner_logits_output = require_component_logits_output(spec, &inner)?;

        // Explicit I/O metadata for each decoder, threaded into decode-state
        // resolution so token/position/KV ports are read from declared metadata
        // instead of an ambiguous tensor-shape guess.
        let outer_io = spec.models.get(&outer).and_then(|model| model.io.clone());
        let inner_io = spec.models.get(&inner).and_then(|model| model.io.clone());

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
            inner_embedding_output,
            outer_logits_output,
            inner_logits_output,
            outer_io,
            inner_io,
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
        if start_step > num_steps {
            anyhow::bail!(
                "iterative strategy 'start_step' ({start_step}) must be <= 'num_steps' ({num_steps})"
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

/// Resolve a pipeline component's explicitly declared logits output port.
///
/// A logits output is shape-ambiguous against other float outputs (hidden
/// states, embeddings), so the nested-AR loop reads the exact port declared in
/// `models.{component}.io.logits_output`. A missing or empty declaration is an
/// actionable error naming the key, never a tensor-name guess.
fn require_component_logits_output(spec: &PipelineSpec, component: &str) -> anyhow::Result<String> {
    let logits_output = spec
        .models
        .get(component)
        .and_then(|model| model.io.as_ref())
        .and_then(|io| io.logits_output.clone())
        .with_context(|| {
            format!(
                "nested_autoregressive decoder '{component}' is missing an explicit logits \
                 output; declare 'models.{component}.io.logits_output'"
            )
        })?;
    if logits_output.is_empty() {
        anyhow::bail!(
            "nested_autoregressive decoder '{component}' declares an empty \
             'models.{component}.io.logits_output'"
        );
    }
    Ok(logits_output)
}

fn component_phase(spec: &PipelineSpec, component: &str, decoder: &str) -> PhaseRunOn {
    // An embeds-driven decoder (`sequence_source: inputs_embeds`) must be fed a
    // fresh embedding for the single running token on every decode step, so the
    // upstream component that produces its `inputs_embeds` runs `every_step`
    // even when the model package labels it `prompt_only`. A prompt-only
    // embedding would run once over the prompt and leave the decoder consuming
    // stale prefill embeddings for every generated token (a shape mismatch at
    // best, silently wrong output at worst). This mirrors the `muse_decode`
    // harness, which re-embeds the running token each step. The reclassification
    // is scoped to the single dataflow producer of the decoder's `inputs_embeds`
    // port, so cached conditioning producers (e.g. a vision encoder feeding the
    // embedder's image features) keep their declared `prompt_only` phase.
    if decoder_embeds_producer(spec, decoder).as_deref() == Some(component) {
        return PhaseRunOn::EveryStep;
    }
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

/// The pipeline component that produces an embeds-driven `decoder`'s
/// `inputs_embeds`, or `None` for a token-id-driven decoder (or when no dataflow
/// edge feeds the embeds port).
///
/// Resolved structurally from metadata: the decoder must declare
/// `sequence_source: inputs_embeds` with an `inputs_embeds_input` port, and some
/// other component must feed that port over a dataflow edge. This is the single
/// upstream embedder whose phase [`component_phase`] promotes to `every_step`.
fn decoder_embeds_producer(spec: &PipelineSpec, decoder: &str) -> Option<String> {
    let io = spec
        .models
        .get(decoder)
        .and_then(|model| model.io.as_ref())?;
    if io.sequence_source != Some(SequenceInputKind::InputsEmbeds) {
        return None;
    }
    let embeds_port = io.inputs_embeds_input.as_deref()?;
    let target = format!("{decoder}.{embeds_port}");
    spec.dataflow
        .iter()
        .find(|edge| edge.to == target)
        .and_then(|edge| endpoint_component(&edge.from))
        .filter(|from| *from != decoder)
        .map(str::to_string)
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

    fn component(role: &str) -> PipelineComponentSpec {
        PipelineComponentSpec {
            filename: format!("{role}.onnx"),
            role: role.to_string(),
            device_preference: None,
            tokenizer: None,
            io: None,
        }
    }

    /// A decoder component whose explicit `io` block is built from a JSON value
    /// (`ModelIoSpec` has no `Default`, and constructing it from JSON keeps the
    /// test focused on just the declared ports).
    fn decoder_with_io(io: serde_json::Value) -> PipelineComponentSpec {
        PipelineComponentSpec {
            filename: "decoder.onnx".to_string(),
            role: "decoder".to_string(),
            device_preference: None,
            tokenizer: None,
            io: Some(serde_json::from_value(io).expect("valid ModelIoSpec JSON")),
        }
    }

    /// A minimal `nested_autoregressive` (multi-decoder TTS) spec with the
    /// required outer→inner per-frame hidden binding. `inner_embedding_output`
    /// is threaded through so both the missing and declared cases can be tested.
    /// Both decoders declare an explicit `io.logits_output`.
    fn nested_autoregressive_spec(inner_embedding_output: Option<String>) -> PipelineSpec {
        PipelineSpec {
            models: BTreeMap::from([
                (
                    "talker".to_string(),
                    decoder_with_io(serde_json::json!({ "logits_output": "talker_logits" })),
                ),
                (
                    "code_predictor".to_string(),
                    decoder_with_io(serde_json::json!({ "logits_output": "code_logits" })),
                ),
            ]),
            dataflow: vec![DataflowEdge {
                from: "talker.last_hidden_state".to_string(),
                to: "code_predictor.inputs_embeds".to_string(),
                dtype: None,
                device_transfer: None,
            }],
            strategy: PipelineStrategy {
                kind: PipelineStrategyKind::NestedAutoregressive,
                outer: Some("talker".to_string()),
                inner: Some("code_predictor".to_string()),
                num_code_groups: Some(4),
                max_tokens: Some(8),
                inner_embedding_output,
                ..PipelineStrategy::default()
            },
            ..PipelineSpec::default()
        }
    }

    #[test]
    fn nested_autoregressive_requires_explicit_inner_embedding_output() {
        let error = PipelinePlan::from_spec(
            &nested_autoregressive_spec(None),
            &SchedulerRegistry::builtin(),
        )
        .expect_err("a nested_autoregressive plan without the inner-embedding contract must fail");
        let message = error.to_string();
        assert!(
            message.contains("inner_embedding_output"),
            "the error must name the missing key: {message}"
        );
    }

    #[test]
    fn nested_autoregressive_rejects_empty_inner_embedding_output() {
        let error = PipelinePlan::from_spec(
            &nested_autoregressive_spec(Some(String::new())),
            &SchedulerRegistry::builtin(),
        )
        .expect_err("an empty inner_embedding_output must fail");
        assert!(
            error.to_string().contains("inner_embedding_output"),
            "the error must name the offending key: {error}"
        );
    }

    #[test]
    fn nested_autoregressive_carries_declared_inner_embedding_output() -> anyhow::Result<()> {
        let plan = PipelinePlan::from_spec(
            &nested_autoregressive_spec(Some("code_embeds".to_string())),
            &SchedulerRegistry::builtin(),
        )?;
        match plan {
            PipelinePlan::NestedAutoregressive(nested) => {
                assert_eq!(nested.inner_embedding_output, "code_embeds");
                assert_eq!(nested.outer_logits_output, "talker_logits");
                assert_eq!(nested.inner_logits_output, "code_logits");
            }
            other => panic!("expected a nested_autoregressive plan, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn nested_autoregressive_requires_explicit_logits_output() {
        // A fully-formed spec whose inner decoder omits its explicit
        // `io.logits_output` must fail closed, naming the missing key rather
        // than falling back to a `"logits"` substring match.
        let mut spec = nested_autoregressive_spec(Some("code_embeds".to_string()));
        spec.models.get_mut("code_predictor").unwrap().io = None;
        let error = PipelinePlan::from_spec(&spec, &SchedulerRegistry::builtin())
            .expect_err("a missing inner logits output must fail");
        let message = error.to_string();
        assert!(
            message.contains("io.logits_output"),
            "the error must name the missing key: {message}"
        );
        assert!(
            message.contains("code_predictor"),
            "the error must name the offending component: {message}"
        );
    }

    #[cfg(feature = "native-backend")]
    #[test]
    fn native_decoder_device_prefers_config_when_env_unset() {
        use crate::native_decode_device::NativeDecodeDevice;
        // Env unset -> honor the engine-configured device (this is the `--ep cuda`
        // fix: the pipeline decoder must run on the configured GPU).
        assert_eq!(
            resolve_native_decoder_device(None, Some(&NativeDecodeDevice::Cuda { index: Some(3) })),
            NativeDecodeDevice::Cuda { index: Some(3) }
        );
        // Env unset and no configured device -> CPU default.
        assert_eq!(
            resolve_native_decoder_device(None, None),
            NativeDecodeDevice::Cpu
        );
    }

    #[cfg(feature = "native-backend")]
    #[test]
    fn native_decoder_device_env_override_wins_over_config() {
        use crate::native_decode_device::NativeDecodeDevice;
        // A set env var wins over the configured device, in both directions, for
        // back-compat with the deterministic parity fixture.
        assert_eq!(
            resolve_native_decoder_device(
                Some("cpu"),
                Some(&NativeDecodeDevice::Cuda { index: Some(1) })
            ),
            NativeDecodeDevice::Cpu
        );
        assert_eq!(
            resolve_native_decoder_device(Some("cuda:2"), Some(&NativeDecodeDevice::Cpu)),
            NativeDecodeDevice::Cuda { index: Some(2) }
        );
        assert_eq!(
            resolve_native_decoder_device(Some("cuda"), None),
            NativeDecodeDevice::Cuda { index: None }
        );
    }

    #[cfg(feature = "native-backend")]
    #[test]
    fn native_decoder_device_value_parsing() {
        use crate::native_decode_device::NativeDecodeDevice;
        assert_eq!(
            parse_native_decoder_device_value("  CUDA=0 "),
            NativeDecodeDevice::Cuda { index: Some(0) }
        );
        assert_eq!(
            parse_native_decoder_device_value("cuda"),
            NativeDecodeDevice::Cuda { index: None }
        );
        // Non-numeric index degrades to the default CUDA device, not CPU.
        assert_eq!(
            parse_native_decoder_device_value("cuda:xyz"),
            NativeDecodeDevice::Cuda { index: None }
        );
        // Any unrecognized value falls back to CPU.
        assert_eq!(
            parse_native_decoder_device_value("gpu"),
            NativeDecodeDevice::Cpu
        );
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
                inner_embedding_output: None,
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
                            inner_embedding_output: None,
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
                            inner_embedding_output: None,
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
            inner_embedding_output: None,
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

//! Compatibility layer that converts a **ComfyUI API-format workflow JSON** into
//! the native onnx-genai [`InferenceMetadata`] spec.
//!
//! ComfyUI is a node-graph UI for diffusion. Its *"Save (API Format)"* export is
//! a flat map `{node_id: {"class_type": str, "inputs": {port: value | link}}}`
//! where a value of the form `[src_id, slot]` is a *link* to another node's
//! output.
//!
//! This crate is the ComfyUI analog of [`onnx-genai-genai-config`]: it parses an
//! external config and produces native [`InferenceMetadata`], carrying only the
//! pipeline *topology + run parameters* — never weights. When the referenced
//! model components already exist as ONNX graphs (`denoiser.onnx`, `vae.onnx`,
//! `text_encoder.onnx`), the onnx-genai iterative pipeline can run the ComfyUI
//! workflow directly, with no Python translation step. In other words the same
//! config runs both in ComfyUI and here, as long as the model itself is ONNX.
//!
//! The canonical text-to-image graph is *KSampler-centric* and maps directly
//! onto onnx-genai's composite iterative pipeline:
//!
//! ```text
//! EmptyLatentImage ─► KSampler ─► VAEDecode ─► SaveImage
//! CLIPTextEncode(+/-) ─┘  (positive / negative → CFG cond / uncond)
//! CheckpointLoaderSimple ─► model / clip / vae
//! ```
//!
//! Only the core txt2img subset is modeled today; unsupported samplers or a
//! missing sampler node raise a clear [`ComfyUiConfigError`] rather than
//! silently producing wrong dynamics. This mirrors the reference Python
//! translator in `mobius.integrations.onnx_genai.comfyui`.

use std::path::Path;

use onnx_genai_metadata::InferenceMetadata;
use serde_json::{Map, Value, json};

/// Default ONNX component filenames the emitted pipeline references.
const DENOISER_FILENAME: &str = "denoiser.onnx";
const VAE_FILENAME: &str = "vae.onnx";
const TEXT_ENCODER_FILENAME: &str = "text_encoder.onnx";

/// Default denoiser I/O port names (match the Mobius exporter contract).
const DENOISER_SAMPLE_INPUT: &str = "sample";
const DENOISER_TIMESTEP_INPUT: &str = "timestep";
const DENOISER_CONDITIONING_INPUT: &str = "encoder_hidden_states";
const DENOISER_OUTPUT: &str = "noise_pred";
const VAE_LATENT_INPUT: &str = "latent";
const TEXT_ENCODER_OUTPUT: &str = "last_hidden_state";

const MAX_TRACE_DEPTH: usize = 16;

/// ComfyUI node `class_type`s the translator recognizes.
const SAMPLER_NODES: &[&str] = &["KSampler", "KSamplerAdvanced"];
const VAE_DECODE_NODES: &[&str] = &["VAEDecode", "VAEDecodeTiled"];
const TEXT_ENCODE_NODES: &[&str] = &["CLIPTextEncode"];
const LATENT_NODES: &[&str] = &["EmptyLatentImage", "EmptySD3LatentImage"];

/// ComfyUI sigma spacings onnx-genai reproduces. `normal`/`simple`/`ddim_uniform`
/// map to linspace; `karras`/`exponential` enable their schedules.
const SUPPORTED_SPACINGS: &[&str] =
    &["normal", "simple", "ddim_uniform", "karras", "exponential"];

/// Errors produced while parsing a ComfyUI workflow.
#[derive(Debug, thiserror::Error)]
pub enum ComfyUiConfigError {
    /// The file could not be read.
    #[error("failed to read ComfyUI workflow: {0}")]
    Io(#[from] std::io::Error),
    /// The file was not valid JSON.
    #[error("failed to parse ComfyUI workflow JSON: {0}")]
    Parse(#[from] serde_json::Error),
    /// The graph was structurally valid JSON but not a supported workflow.
    #[error("unsupported ComfyUI workflow: {0}")]
    Unsupported(String),
}

/// Map a ComfyUI `sampler_name` to an onnx-genai scheduler `kind`.
///
/// Only samplers with an onnx-genai implementation are mapped; others are
/// rejected until onnx-genai grows an equivalent scheduler.
fn sampler_kind(sampler_name: &str) -> Result<&'static str, ComfyUiConfigError> {
    Ok(match sampler_name {
        "euler" => "euler",
        "euler_ancestral" => "euler_ancestral",
        "ddim" => "ddim",
        "dpmpp_2m" | "dpm_2m" => "dpmpp_2m",
        other => {
            return Err(ComfyUiConfigError::Unsupported(format!(
                "ComfyUI sampler {other:?} has no onnx-genai equivalent yet; supported: \
                 ddim, dpmpp_2m, euler, euler_ancestral"
            )));
        }
    })
}

/// Diffusion noise-schedule parameters for an onnx-genai scheduler.
///
/// Mirrors `SchedulerConfig` in the reference Python translator. Defaults are
/// the Stable Diffusion values.
#[derive(Debug, Clone, PartialEq)]
pub struct SchedulerConfig {
    /// Scheduler algorithm (`ddim`, `euler`, `euler_ancestral`, `dpmpp_2m`).
    pub kind: String,
    /// Training timesteps the schedule was defined over.
    pub num_train_timesteps: usize,
    /// Linear beta-schedule start.
    pub beta_start: f64,
    /// Linear beta-schedule end.
    pub beta_end: f64,
    /// Beta schedule shape (`scaled_linear` for Stable Diffusion).
    pub beta_schedule: String,
    /// Model output parameterization.
    pub prediction_type: String,
    /// Use the Karras sigma spacing.
    pub use_karras_sigmas: bool,
    /// Use the exponential sigma spacing.
    pub use_exponential_sigmas: bool,
}

impl SchedulerConfig {
    /// Build a config for `kind` with Stable Diffusion schedule defaults.
    pub fn new(kind: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            num_train_timesteps: 1000,
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: "scaled_linear".into(),
            prediction_type: "epsilon".into(),
            use_karras_sigmas: false,
            use_exponential_sigmas: false,
        }
    }

    fn to_metadata_value(&self) -> Value {
        let mut meta = Map::new();
        meta.insert("kind".into(), json!(self.kind));
        meta.insert(
            "num_train_timesteps".into(),
            json!(self.num_train_timesteps),
        );
        meta.insert("beta_start".into(), json!(self.beta_start));
        meta.insert("beta_end".into(), json!(self.beta_end));
        meta.insert("beta_schedule".into(), json!(self.beta_schedule));
        meta.insert("prediction_type".into(), json!(self.prediction_type));
        if self.use_karras_sigmas {
            meta.insert("use_karras_sigmas".into(), json!(true));
        }
        if self.use_exponential_sigmas {
            meta.insert("use_exponential_sigmas".into(), json!(true));
        }
        Value::Object(meta)
    }
}

/// Everything needed to run a translated ComfyUI txt2img workflow.
///
/// `metadata` is the native onnx-genai pipeline document (topology + scheduler +
/// guidance). The remaining fields are the per-run inputs recovered from the
/// graph so a caller (native runner or Python driver) can actually drive it.
#[derive(Debug, Clone)]
pub struct ComfyUiWorkflow {
    /// Native pipeline metadata (topology + scheduler + guidance).
    pub metadata: InferenceMetadata,
    /// Positive-conditioning prompt text, if recoverable.
    pub prompt: Option<String>,
    /// Negative-conditioning prompt text, if recoverable.
    pub negative_prompt: Option<String>,
    /// Latent width in pixels.
    pub width: u32,
    /// Latent height in pixels.
    pub height: u32,
    /// Number of images to generate.
    pub batch_size: u32,
    /// RNG seed.
    pub seed: i64,
    /// Number of denoise steps.
    pub steps: u32,
    /// Classifier-free guidance scale (1.0 disables CFG).
    pub cfg: f64,
    /// The raw ComfyUI `sampler_name`.
    pub sampler_name: String,
    /// The mapped onnx-genai scheduler kind.
    pub scheduler_kind: String,
    /// The raw ComfyUI `scheduler` spacing.
    pub scheduler_spacing: String,
    /// Referenced checkpoint / UNet filename, if recoverable.
    pub checkpoint: Option<String>,
    /// KSampler `denoise` strength (< 1.0 => partial img2img loop).
    pub denoise: f64,
    /// First step index for a partial denoise loop.
    pub start_step: u32,
    /// LoRA `(name, strength)` pairs along the model chain, in application order.
    pub loras: Vec<(String, f64)>,
    /// ControlNet `(name, strength)` if present.
    pub controlnet: Option<(String, f64)>,
}

/// Return the flat node map, tolerating a `{"prompt": {...}}` wrapper.
fn nodes_of(workflow: &Value) -> Option<&Map<String, Value>> {
    let obj = workflow.as_object()?;
    if let Some(Value::Object(inner)) = obj.get("prompt") {
        return Some(inner);
    }
    Some(obj)
}

/// A `[src_id, slot]` link references another node's output.
fn as_link(value: &Value) -> Option<&str> {
    let arr = value.as_array()?;
    if arr.len() == 2 && arr[0].is_string() && arr[1].is_i64() {
        arr[0].as_str()
    } else {
        None
    }
}

/// Follow a `[src_id, slot]` link to the referenced node object.
fn resolve<'a>(nodes: &'a Map<String, Value>, reference: Option<&Value>) -> Option<&'a Value> {
    let src = as_link(reference?)?;
    nodes.get(src).filter(|n| n.is_object())
}

fn node_class(node: &Value) -> Option<&str> {
    node.get("class_type").and_then(Value::as_str)
}

fn node_inputs(node: &Value) -> Option<&Map<String, Value>> {
    node.get("inputs").and_then(Value::as_object)
}

fn input<'a>(node: &'a Value, key: &str) -> Option<&'a Value> {
    node_inputs(node)?.get(key)
}

/// Find the single node of one of `class_types`; error if absent or ambiguous.
fn find_single<'a>(
    nodes: &'a Map<String, Value>,
    class_types: &[&str],
    what: &str,
) -> Result<&'a Value, ComfyUiConfigError> {
    let mut hits = nodes
        .values()
        .filter(|n| node_class(n).is_some_and(|c| class_types.contains(&c)));
    let first = hits.next().ok_or_else(|| {
        ComfyUiConfigError::Unsupported(format!(
            "ComfyUI workflow has no {what} node ({}); this translator supports the core \
             text-to-image (KSampler) graph",
            class_types.join(" / ")
        ))
    })?;
    if hits.next().is_some() {
        return Err(ComfyUiConfigError::Unsupported(format!(
            "ComfyUI workflow has multiple {what} nodes; only one is supported"
        )));
    }
    Ok(first)
}

/// Resolve a KSampler conditioning link to its CLIPTextEncode prompt text.
fn follow_prompt_text(
    nodes: &Map<String, Value>,
    reference: Option<&Value>,
    depth: usize,
) -> Option<String> {
    if depth == 0 {
        return None;
    }
    let node = resolve(nodes, reference)?;
    if node_class(node).is_some_and(|c| TEXT_ENCODE_NODES.contains(&c)) {
        return input(node, "text")
            .and_then(Value::as_str)
            .map(str::to_owned);
    }
    // Some graphs wrap conditioning (e.g. ConditioningCombine); follow one hop.
    let inner = input(node, "conditioning")?;
    if as_link(inner).is_some() {
        return follow_prompt_text(nodes, Some(inner), depth - 1);
    }
    None
}

/// Resolve a KSampler latent link to `(width, height, batch_size)`.
fn follow_dims(nodes: &Map<String, Value>, reference: Option<&Value>) -> (u32, u32, u32) {
    if let Some(node) = resolve(nodes, reference)
        && node_class(node).is_some_and(|c| LATENT_NODES.contains(&c))
    {
        let width = input(node, "width").and_then(Value::as_u64).unwrap_or(512) as u32;
        let height = input(node, "height").and_then(Value::as_u64).unwrap_or(512) as u32;
        let batch = input(node, "batch_size")
            .and_then(Value::as_u64)
            .unwrap_or(1)
            .max(1) as u32;
        return (width, height, batch);
    }
    (512, 512, 1)
}

/// Trace a KSampler.model link back to a checkpoint/UNet filename.
fn trace_checkpoint(nodes: &Map<String, Value>, reference: Option<&Value>) -> Option<String> {
    let mut reference = reference;
    for _ in 0..MAX_TRACE_DEPTH {
        let node = resolve(nodes, reference)?;
        for key in ["ckpt_name", "unet_name", "model_name"] {
            if let Some(name) = input(node, key).and_then(Value::as_str) {
                return Some(name.to_owned());
            }
        }
        reference = input(node, "model");
        as_link(reference?)?;
    }
    None
}

/// Collect LoraLoader nodes along a KSampler.model chain, in application order.
fn trace_loras(nodes: &Map<String, Value>, reference: Option<&Value>) -> Vec<(String, f64)> {
    let mut loras = Vec::new();
    let mut reference = reference;
    for _ in 0..MAX_TRACE_DEPTH {
        let Some(node) = resolve(nodes, reference) else {
            break;
        };
        if node_class(node).is_some_and(|c| matches!(c, "LoraLoader" | "LoraLoaderModelOnly"))
            && let Some(name) = input(node, "lora_name").and_then(Value::as_str)
        {
            let strength = input(node, "strength_model")
                .or_else(|| input(node, "strength"))
                .and_then(Value::as_f64)
                .unwrap_or(1.0);
            loras.push((name.to_owned(), strength));
        }
        reference = input(node, "model");
        if reference.and_then(as_link).is_none() {
            break;
        }
    }
    loras.reverse(); // base checkpoint applies first
    loras
}

/// Find a ControlNetApply node's `(control_net_name, strength)`, if any.
fn find_controlnet(nodes: &Map<String, Value>) -> Option<(String, f64)> {
    for node in nodes.values() {
        if node_class(node)
            .is_some_and(|c| matches!(c, "ControlNetApply" | "ControlNetApplyAdvanced"))
        {
            let strength = input(node, "strength").and_then(Value::as_f64).unwrap_or(1.0);
            let loader = resolve(nodes, input(node, "control_net"));
            if let Some(loader) = loader
                && node_class(loader)
                    .is_some_and(|c| matches!(c, "ControlNetLoader" | "DiffControlNetLoader"))
                && let Some(name) = input(loader, "control_net_name").and_then(Value::as_str)
            {
                return Some((name.to_owned(), strength));
            }
        }
    }
    None
}

/// Build the native pipeline metadata document for a diffusion workflow.
///
/// Mirrors `build_diffusion_pipeline_metadata` in the reference Python module:
/// the denoiser runs an iterative loop (`noise_pred -> sample` self-edge), the
/// per-step timestep is injected into `timestep`, an optional text encoder runs
/// `prompt_only` feeding conditioning, and an optional VAE runs `final_only`.
#[allow(clippy::too_many_arguments)]
fn build_pipeline_metadata(
    num_steps: u32,
    scheduler: &SchedulerConfig,
    guidance_scale: Option<f64>,
    start_step: Option<u32>,
    has_text_encoder: bool,
    has_vae: bool,
) -> Result<InferenceMetadata, ComfyUiConfigError> {
    if num_steps < 1 {
        return Err(ComfyUiConfigError::Unsupported(
            "num_steps must be >= 1".into(),
        ));
    }

    let mut models = Map::new();
    models.insert(
        "denoiser".into(),
        json!({ "filename": DENOISER_FILENAME, "type": "denoiser" }),
    );

    let mut dataflow = vec![json!({
        "from": format!("denoiser.{DENOISER_OUTPUT}"),
        "to": format!("denoiser.{DENOISER_SAMPLE_INPUT}"),
    })];
    let mut phases = Map::new();

    if has_text_encoder {
        models.insert(
            "text_encoder".into(),
            json!({ "filename": TEXT_ENCODER_FILENAME, "type": "encoder" }),
        );
        dataflow.push(json!({
            "from": format!("text_encoder.{TEXT_ENCODER_OUTPUT}"),
            "to": format!("denoiser.{DENOISER_CONDITIONING_INPUT}"),
        }));
        phases.insert("text_encoder".into(), json!({ "run_on": "prompt_only" }));
    }

    if has_vae {
        models.insert(
            "vae".into(),
            json!({ "filename": VAE_FILENAME, "type": "vae" }),
        );
        dataflow.push(json!({
            "from": format!("denoiser.{DENOISER_SAMPLE_INPUT}"),
            "to": format!("vae.{VAE_LATENT_INPUT}"),
        }));
        phases.insert("vae".into(), json!({ "run_on": "final_only" }));
    }

    let mut strategy = Map::new();
    strategy.insert("kind".into(), json!("iterative"));
    strategy.insert("denoiser".into(), json!("denoiser"));
    strategy.insert("num_steps".into(), json!(num_steps));
    strategy.insert("timestep_input".into(), json!(DENOISER_TIMESTEP_INPUT));
    strategy.insert(
        "scheduler_config".into(),
        scheduler.to_metadata_value(),
    );
    if let Some(scale) = guidance_scale {
        strategy.insert("guidance_scale".into(), json!(scale));
        if scale != 1.0 {
            strategy.insert(
                "cfg_conditioning_input".into(),
                json!(DENOISER_CONDITIONING_INPUT),
            );
        }
    }
    if let Some(start) = start_step {
        if start == 0 || start >= num_steps {
            return Err(ComfyUiConfigError::Unsupported(format!(
                "start_step ({start}) must be in 1..{}",
                num_steps - 1
            )));
        }
        strategy.insert("start_step".into(), json!(start));
    }

    let mut pipeline = Map::new();
    pipeline.insert("models".into(), Value::Object(models));
    pipeline.insert("dataflow".into(), Value::Array(dataflow));
    pipeline.insert("strategy".into(), Value::Object(strategy));
    if !phases.is_empty() {
        pipeline.insert("phases".into(), Value::Object(phases));
    }

    let root = json!({ "pipeline": Value::Object(pipeline) });
    Ok(serde_json::from_value(root)?)
}

/// Parse a ComfyUI API-format workflow (`serde_json::Value`) into a structured
/// [`ComfyUiWorkflow`] carrying native metadata + run parameters.
pub fn parse_workflow(workflow: &Value) -> Result<ComfyUiWorkflow, ComfyUiConfigError> {
    let nodes = nodes_of(workflow).ok_or_else(|| {
        ComfyUiConfigError::Unsupported("workflow is not a JSON object of nodes".into())
    })?;

    let sampler = find_single(nodes, SAMPLER_NODES, "sampler")?;

    let steps = input(sampler, "steps")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            ComfyUiConfigError::Unsupported("ComfyUI sampler node is missing 'steps'".into())
        })? as u32;
    let cfg = input(sampler, "cfg").and_then(Value::as_f64).unwrap_or(1.0);
    let sampler_name = input(sampler, "sampler_name")
        .and_then(Value::as_str)
        .unwrap_or("euler")
        .to_owned();
    let spacing = input(sampler, "scheduler")
        .and_then(Value::as_str)
        .unwrap_or("normal")
        .to_owned();
    let seed = input(sampler, "seed")
        .or_else(|| input(sampler, "noise_seed"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let denoise = input(sampler, "denoise")
        .and_then(Value::as_f64)
        .unwrap_or(1.0);

    let kind = sampler_kind(&sampler_name)?;
    let known_spacing = SUPPORTED_SPACINGS.contains(&spacing.as_str());

    // img2img: a KSampler `denoise` < 1.0 skips the earliest (noisiest) steps.
    // start_step = num_steps - round(num_steps * denoise). Uses banker's rounding
    // to match numpy/diffusers get_timesteps.
    let mut start_step = 0u32;
    if denoise > 0.0 && denoise < 1.0 {
        let raw = (steps as f64) * denoise;
        let rounded = round_ties_even(raw) as i64;
        let candidate = steps as i64 - rounded;
        start_step = candidate.clamp(0, (steps as i64 - 1).max(0)) as u32;
    }

    let prompt = follow_prompt_text(nodes, input(sampler, "positive"), MAX_TRACE_DEPTH);
    let negative_prompt = follow_prompt_text(nodes, input(sampler, "negative"), MAX_TRACE_DEPTH);
    let (width, height, batch_size) = follow_dims(nodes, input(sampler, "latent_image"));
    let checkpoint = trace_checkpoint(nodes, input(sampler, "model"));
    let loras = trace_loras(nodes, input(sampler, "model"));
    let controlnet = find_controlnet(nodes);

    let has_text_encoder = nodes
        .values()
        .any(|n| node_class(n).is_some_and(|c| TEXT_ENCODE_NODES.contains(&c)));
    let has_vae = nodes
        .values()
        .any(|n| node_class(n).is_some_and(|c| VAE_DECODE_NODES.contains(&c)));
    let guidance = if cfg != 1.0 { Some(cfg) } else { None };

    let scheduler = SchedulerConfig {
        use_karras_sigmas: spacing == "karras",
        use_exponential_sigmas: spacing == "exponential",
        ..SchedulerConfig::new(kind)
    };
    // Unknown spacings still parse; the runtime falls back to linspace. We keep
    // the flag off so behavior is well-defined (parity with the Python warning).
    let _ = known_spacing;

    let metadata = build_pipeline_metadata(
        steps,
        &scheduler,
        guidance,
        if start_step > 0 { Some(start_step) } else { None },
        has_text_encoder,
        has_vae,
    )?;

    Ok(ComfyUiWorkflow {
        metadata,
        prompt,
        negative_prompt,
        width,
        height,
        batch_size,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler_kind: scheduler.kind,
        scheduler_spacing: spacing,
        checkpoint,
        denoise,
        start_step,
        loras,
        controlnet,
    })
}

/// Parse a ComfyUI API-format workflow from a JSON string.
pub fn parse_workflow_str(json: &str) -> Result<ComfyUiWorkflow, ComfyUiConfigError> {
    let value: Value = serde_json::from_str(json)?;
    parse_workflow(&value)
}

/// Load and parse a ComfyUI API-format workflow JSON file.
pub fn parse_workflow_file(path: impl AsRef<Path>) -> Result<ComfyUiWorkflow, ComfyUiConfigError> {
    let text = std::fs::read_to_string(path)?;
    parse_workflow_str(&text)
}

/// Round half to even (banker's rounding), matching numpy `round`.
fn round_ties_even(x: f64) -> f64 {
    let rounded = x.round();
    if (x - x.trunc()).abs() == 0.5 {
        // Exactly halfway: round to the nearest even integer.
        let floor = x.floor();
        if (floor as i64) % 2 == 0 {
            floor
        } else {
            floor + 1.0
        }
    } else {
        rounded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn txt2img_graph() -> Value {
        json!({
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": 42,
                    "steps": 20,
                    "cfg": 7.5,
                    "sampler_name": "euler",
                    "scheduler": "karras",
                    "denoise": 1.0,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0]
                }
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "sd15.safetensors"}
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": 768, "height": 512, "batch_size": 2}
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "a fox", "clip": ["4", 1]}
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "blurry", "clip": ["4", 1]}
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["3", 0], "vae": ["4", 2]}
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {"images": ["8", 0]}
            }
        })
    }

    #[test]
    fn parses_core_txt2img() {
        let workflow = parse_workflow(&txt2img_graph()).unwrap();
        assert_eq!(workflow.prompt.as_deref(), Some("a fox"));
        assert_eq!(workflow.negative_prompt.as_deref(), Some("blurry"));
        assert_eq!((workflow.width, workflow.height, workflow.batch_size), (768, 512, 2));
        assert_eq!(workflow.seed, 42);
        assert_eq!(workflow.steps, 20);
        assert_eq!(workflow.cfg, 7.5);
        assert_eq!(workflow.sampler_name, "euler");
        assert_eq!(workflow.scheduler_kind, "euler");
        assert_eq!(workflow.scheduler_spacing, "karras");
        assert_eq!(workflow.checkpoint.as_deref(), Some("sd15.safetensors"));
        assert_eq!(workflow.start_step, 0);
    }

    #[test]
    fn builds_iterative_pipeline_metadata() {
        let workflow = parse_workflow(&txt2img_graph()).unwrap();
        let pipeline = workflow.metadata.pipeline.expect("pipeline present");
        // Three components: denoiser, text_encoder, vae.
        assert!(pipeline.models.contains_key("denoiser"));
        assert!(pipeline.models.contains_key("text_encoder"));
        assert!(pipeline.models.contains_key("vae"));
        let strategy = &pipeline.strategy;
        assert_eq!(strategy.num_steps, Some(20));
        assert_eq!(strategy.denoiser.as_deref(), Some("denoiser"));
        assert_eq!(strategy.timestep_input.as_deref(), Some("timestep"));
        // CFG on (cfg=7.5 != 1.0).
        assert_eq!(strategy.guidance_scale, Some(7.5));
        assert_eq!(
            strategy.cfg_conditioning_input.as_deref(),
            Some("encoder_hidden_states")
        );
        // Karras spacing propagated to the scheduler config.
        let scheduler = strategy.scheduler_config.as_ref().unwrap();
        assert_eq!(scheduler.kind, "euler");
        assert_eq!(scheduler.use_karras_sigmas, Some(true));
        // Loop-carried self-edge is present.
        assert!(
            pipeline
                .dataflow
                .iter()
                .any(|e| e.from == "denoiser.noise_pred" && e.to == "denoiser.sample")
        );
    }

    #[test]
    fn cfg_one_disables_guidance() {
        let mut graph = txt2img_graph();
        graph["3"]["inputs"]["cfg"] = json!(1.0);
        let workflow = parse_workflow(&graph).unwrap();
        let strategy = workflow.metadata.pipeline.unwrap().strategy;
        assert_eq!(strategy.guidance_scale, None);
        assert_eq!(strategy.cfg_conditioning_input, None);
    }

    #[test]
    fn denoise_below_one_sets_start_step() {
        let mut graph = txt2img_graph();
        graph["3"]["inputs"]["denoise"] = json!(0.6);
        let workflow = parse_workflow(&graph).unwrap();
        // 20 - round(20*0.6) = 20 - 12 = 8.
        assert_eq!(workflow.start_step, 8);
        assert_eq!(
            workflow.metadata.pipeline.unwrap().strategy.start_step,
            Some(8)
        );
    }

    #[test]
    fn traces_loras_in_application_order() {
        let mut graph = txt2img_graph();
        // KSampler.model -> LoraLoader(b) -> LoraLoader(a) -> checkpoint.
        graph["3"]["inputs"]["model"] = json!(["10", 0]);
        graph["10"] = json!({
            "class_type": "LoraLoader",
            "inputs": {"lora_name": "b.safetensors", "strength_model": 0.5, "model": ["11", 0]}
        });
        graph["11"] = json!({
            "class_type": "LoraLoader",
            "inputs": {"lora_name": "a.safetensors", "strength_model": 0.8, "model": ["4", 0]}
        });
        let workflow = parse_workflow(&graph).unwrap();
        assert_eq!(
            workflow.loras,
            vec![
                ("a.safetensors".to_owned(), 0.8),
                ("b.safetensors".to_owned(), 0.5),
            ]
        );
        assert_eq!(workflow.checkpoint.as_deref(), Some("sd15.safetensors"));
    }

    #[test]
    fn finds_controlnet() {
        let mut graph = txt2img_graph();
        graph["20"] = json!({
            "class_type": "ControlNetApply",
            "inputs": {"strength": 0.9, "control_net": ["21", 0], "conditioning": ["6", 0]}
        });
        graph["21"] = json!({
            "class_type": "ControlNetLoader",
            "inputs": {"control_net_name": "canny.safetensors"}
        });
        let workflow = parse_workflow(&graph).unwrap();
        assert_eq!(workflow.controlnet, Some(("canny.safetensors".to_owned(), 0.9)));
    }

    #[test]
    fn unwraps_prompt_wrapper() {
        let graph = json!({ "prompt": txt2img_graph() });
        let workflow = parse_workflow(&graph).unwrap();
        assert_eq!(workflow.prompt.as_deref(), Some("a fox"));
    }

    #[test]
    fn rejects_unknown_sampler() {
        let mut graph = txt2img_graph();
        graph["3"]["inputs"]["sampler_name"] = json!("dpm_adaptive");
        let err = parse_workflow(&graph).unwrap_err();
        assert!(matches!(err, ComfyUiConfigError::Unsupported(_)));
    }

    #[test]
    fn rejects_missing_sampler() {
        let graph = json!({
            "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "hi"}}
        });
        assert!(parse_workflow(&graph).is_err());
    }

    #[test]
    fn defaults_dims_without_latent_node() {
        let mut graph = txt2img_graph();
        // Point latent_image at a non-latent node.
        graph["3"]["inputs"]["latent_image"] = json!(["4", 0]);
        let workflow = parse_workflow(&graph).unwrap();
        assert_eq!((workflow.width, workflow.height, workflow.batch_size), (512, 512, 1));
    }

    #[test]
    fn no_vae_or_text_encoder_omits_components() {
        let graph = json!({
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": 0, "steps": 4, "cfg": 1.0, "sampler_name": "ddim",
                    "scheduler": "normal", "model": ["4", 0], "latent_image": ["5", 0]
                }
            },
            "4": {"class_type": "UNETLoader", "inputs": {"unet_name": "unet.onnx"}},
            "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 64, "height": 64}}
        });
        let workflow = parse_workflow(&graph).unwrap();
        let pipeline = workflow.metadata.pipeline.unwrap();
        assert!(pipeline.models.contains_key("denoiser"));
        assert!(!pipeline.models.contains_key("vae"));
        assert!(!pipeline.models.contains_key("text_encoder"));
        assert_eq!(workflow.checkpoint.as_deref(), Some("unet.onnx"));
    }

    #[test]
    fn emitted_metadata_passes_runtime_validator() {
        use onnx_genai_metadata::validate_pipeline_spec;
        let workflow = parse_workflow(&txt2img_graph()).unwrap();
        let pipeline = workflow.metadata.pipeline.expect("pipeline present");
        // The real "can we run it" gate: the runtime pipeline validator accepts
        // the emitted spec (loop-carried denoiser self-edge, single producers,
        // acyclic across components).
        validate_pipeline_spec(&pipeline).expect("emitted pipeline must be runnable");
    }
}

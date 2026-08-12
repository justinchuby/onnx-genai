// Copyright (c) Microsoft Corporation.
//
//! Text-to-image rendering for declarative diffusion pipelines.
//!
//! This module turns a prompt plus sampling parameters into RGB images by
//! driving a `kind: iterative` pipeline package (see [`docs/genai/DIFFUSION.md`]):
//!
//! ```text
//! text_encoder (prompt_only) -> denoiser (iterative) -> vae (final_only)
//! ```
//!
//! Everything the renderer needs is read from the package's declared pipeline
//! metadata — the denoiser component, its loop-carried latent port, the
//! classifier-free-guidance conditioning port, the prompt-phase encoder, and
//! the final-phase component that emits the image. No model family, vendor, or
//! architecture name is hardcoded (see `RULES.md` §2).
//!
//! Packages whose pipeline stops at the latent (no final VAE phase) are
//! supported through [`VaeDecoder`], which decodes the final latent with a
//! standalone ONNX session.
//!
//! [`docs/genai/DIFFUSION.md`]: https://github.com/justinchuby/onnx-genai/blob/main/docs/genai/DIFFUSION.md

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use onnx_genai_engine::{
    GeneratePrompt, GenerateRequest, IterativeOverrides, PipelineEngine, PipelineGenerateRequest,
};
use onnx_genai_metadata::{PhaseRunOn, PipelineSpec};
use onnx_genai_ort::{DataType, Environment, Session, SessionOptions, Value};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use tokenizers::{
    PaddingDirection, PaddingParams, PaddingStrategy, TruncationDirection, TruncationParams,
    TruncationStrategy,
};

/// CLIP context length: prompts are tokenized to exactly this many ids (fixed
/// padding + truncation), matching diffusers `CLIPTokenizer`.
pub const CLIP_CONTEXT_LENGTH: usize = 77;

/// CLIP end-of-text token id. diffusers pads to `max_length` with this token
/// (not id 0), so a runner must match it to reproduce the same ids.
pub const CLIP_END_OF_TEXT_ID: u32 = 49407;

/// VAE spatial downsampling factor (latent side = image side / 8).
pub const VAE_DOWNSCALE: usize = 8;

/// Default latent channel count for a classic Stable Diffusion VAE, used when
/// the package's `run.json` does not declare `latent_channels`.
pub const DEFAULT_LATENT_CHANNELS: usize = 4;

/// Largest batch one render may produce, bounding a single request's cost.
///
/// Owned here rather than by each front end so the CLI and the HTTP API accept
/// exactly the same range.
pub const MAX_BATCH_SIZE: usize = 4;

/// Largest image side this renderer will attempt, in pixels.
///
/// The latent buffer is allocated from width x height before any model runs, so
/// an unbounded size is an allocation an untrusted caller controls.
pub const MAX_IMAGE_SIDE: usize = 4_096;

/// Largest denoise loop this renderer will run.
///
/// Each step is a full denoiser pass, so an unbounded count is unbounded
/// compute for a single request.
pub const MAX_STEPS: usize = 1_000;

/// Validate a requested image size against [`MAX_IMAGE_SIDE`].
pub fn validate_image_size(width: usize, height: usize) -> Result<()> {
    for (axis, value) in [("width", width), ("height", height)] {
        if value == 0 || value > MAX_IMAGE_SIDE {
            bail!(
                "What: an image {axis} of {value} was rejected. \
                 Why: this renderer allocates the latent buffer up front, so each side must be between 1 and {MAX_IMAGE_SIDE} pixels. \
                 How: request a {axis} within that range."
            );
        }
    }
    Ok(())
}

/// Validate a requested step count against [`MAX_STEPS`].
pub fn validate_steps(steps: usize) -> Result<()> {
    if steps == 0 || steps > MAX_STEPS {
        bail!(
            "What: a denoise loop of {steps} steps was rejected. \
             Why: every step is a full denoiser pass, so the count must be between 1 and {MAX_STEPS}. \
             How: request a step count within that range."
        );
    }
    Ok(())
}

/// Validate a requested batch size against [`MAX_BATCH_SIZE`].
pub fn validate_batch_size(batch_size: usize) -> Result<()> {
    if batch_size == 0 || batch_size > MAX_BATCH_SIZE {
        bail!(
            "What: a batch of {batch_size} images was rejected. \
             Why: each render produces between 1 and {MAX_BATCH_SIZE} images. \
             How: request a batch between 1 and {MAX_BATCH_SIZE}."
        );
    }
    Ok(())
}

/// Reject non-finite values produced by an image decode stage.
///
/// Widening an fp16 output to f32 preserves NaN and infinity, so callers must
/// validate after conversion rather than allowing corrupt pixels to reach an
/// encoder or API response.
pub fn validate_finite_decode_output(values: &[f32], stage: &str) -> Result<()> {
    let mut nan_count = 0usize;
    let mut positive_infinity_count = 0usize;
    let mut negative_infinity_count = 0usize;

    for &value in values {
        if value.is_nan() {
            nan_count += 1;
        } else if value == f32::INFINITY {
            positive_infinity_count += 1;
        } else if value == f32::NEG_INFINITY {
            negative_infinity_count += 1;
        }
    }

    let non_finite_count = nan_count + positive_infinity_count + negative_infinity_count;
    if non_finite_count > 0 {
        bail!(
            "What: the {stage} produced {non_finite_count} non-finite values \
             (NaN: {nan_count}, +Inf: {positive_infinity_count}, -Inf: {negative_infinity_count}). \
             Why: fp16 decode overflow can produce NaN or infinity, and widening those values to f32 cannot recover the image. \
             How: export or run this decode stage in fp32 (for example, use an fp32 VAE decoder) and retry."
        );
    }

    Ok(())
}

/// Standalone VAE decoder for packages whose pipeline ends at the latent.
#[derive(Debug, Clone)]
pub struct VaeDecoder {
    /// ONNX model file implementing `latent -> image`.
    pub model_path: PathBuf,
    /// The final latent is divided by this value before decoding. Classic
    /// Stable Diffusion 1.x uses `0.18215`.
    pub scaling_factor: f32,
}

/// Standalone VAE encoder used by img2img and inpainting workflows.
#[derive(Debug, Clone)]
pub struct VaeEncoder {
    /// ONNX model file implementing `image -> latent` (or latent moments).
    pub model_path: PathBuf,
    /// Scale applied to the encoded latent before diffusion.
    pub scaling_factor: f32,
}

/// Source pixels and optional repaint mask for image-conditioned diffusion.
#[derive(Debug, Clone)]
pub struct SourceImage {
    pub width: usize,
    pub height: usize,
    /// RGB pixels in channel-major `[-1, 1]` form.
    pub pixels_chw: Vec<f32>,
    /// Optional single-channel mask in `[0, 1]`; one means repaint.
    pub mask: Option<Vec<f32>>,
}

/// One additional f32 input supplied directly to the declared denoiser.
///
/// This carries package-defined conditioning such as ControlNet images/scales
/// and runtime LoRA gates without adding model-family logic to the engine.
#[derive(Debug, Clone, PartialEq)]
pub struct DenoiserInput {
    /// Denoiser graph input port, without the component prefix.
    pub name: String,
    /// Tensor values in row-major order.
    pub values: Vec<f32>,
    /// Tensor shape; an empty shape denotes a scalar.
    pub shape: Vec<i64>,
}

/// Sampling parameters for one text-to-image render.
#[derive(Debug, Clone)]
pub struct TextToImageRequest {
    /// Positive prompt.
    pub prompt: String,
    /// Negative prompt, encoded as the classifier-free-guidance unconditional
    /// embedding. An empty string reproduces diffusers' default.
    pub negative_prompt: String,
    /// Number of denoise steps. `None` keeps the package's declared `num_steps`.
    pub steps: Option<usize>,
    /// Classifier-free-guidance scale; `1.0` disables guidance. `None` keeps the
    /// package's declared `guidance_scale`.
    pub guidance_scale: Option<f32>,
    /// First denoise step (img2img partial loops). `None` keeps the declared value.
    pub start_step: Option<usize>,
    /// Seed for the initial latent (and, for ancestral schedulers, the per-step noise).
    pub seed: u64,
    /// Output image height in pixels; must be a multiple of [`VAE_DOWNSCALE`].
    pub height: usize,
    /// Output image width in pixels; must be a multiple of [`VAE_DOWNSCALE`].
    pub width: usize,
    /// Number of images rendered in one batch.
    pub batch_size: usize,
    /// CLIP `tokenizer.json`; defaults to `<pipeline_dir>/tokenizer.json`.
    pub tokenizer_path: Option<PathBuf>,
    /// Text-encoder ONNX file used to encode the negative prompt for the
    /// unconditional pass; defaults to `<pipeline_dir>/<encoder filename>`.
    pub text_encoder_path: Option<PathBuf>,
    /// Standalone VAE decode for packages without a final image phase.
    pub vae_decoder: Option<VaeDecoder>,
    /// Source image for img2img/inpainting. `None` selects txt2img.
    pub source_image: Option<SourceImage>,
    /// VAE encoder required when `source_image` is present.
    pub vae_encoder: Option<VaeEncoder>,
}

impl Default for TextToImageRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: String::new(),
            steps: None,
            guidance_scale: None,
            start_step: None,
            seed: 0,
            height: 512,
            width: 512,
            batch_size: 1,
            tokenizer_path: None,
            text_encoder_path: None,
            vae_decoder: None,
            source_image: None,
            vae_encoder: None,
        }
    }
}

/// Load an RGB source image and optional grayscale repaint mask.
pub fn load_source_image(path: &Path, mask_path: Option<&Path>) -> Result<SourceImage> {
    let image = image::open(path)
        .with_context(|| format!("loading source image {}", path.display()))?
        .to_rgb8();
    let (width, height) = image.dimensions();
    let plane = width as usize * height as usize;
    let mut pixels_chw = vec![0.0f32; 3 * plane];
    for (index, pixel) in image.pixels().enumerate() {
        for channel in 0..3 {
            pixels_chw[channel * plane + index] = pixel[channel] as f32 / 127.5 - 1.0;
        }
    }
    let mask = mask_path
        .map(|mask_path| -> Result<Vec<f32>> {
            if mask_path == path {
                let rgba = image::open(path)
                    .with_context(|| format!("loading source alpha mask {}", path.display()))?
                    .to_rgba8();
                return Ok(rgba
                    .pixels()
                    .map(|pixel| 1.0 - pixel[3] as f32 / 255.0)
                    .collect());
            }
            let mask = image::open(mask_path)
                .with_context(|| format!("loading inpainting mask {}", mask_path.display()))?
                .resize_exact(width, height, image::imageops::FilterType::Nearest)
                .to_luma8();
            Ok(mask.pixels().map(|pixel| pixel[0] as f32 / 255.0).collect())
        })
        .transpose()?;
    Ok(SourceImage {
        width: width as usize,
        height: height as usize,
        pixels_chw,
        mask,
    })
}

/// One rendered image as `[3, height, width]` f32 channel-major data in `[-1, 1]`.
#[derive(Debug, Clone)]
pub struct RenderedImage {
    pub width: usize,
    pub height: usize,
    pub pixels_chw: Vec<f32>,
}

/// Endpoints resolved from a package's declared pipeline metadata.
#[derive(Debug, Clone)]
struct DiffusionEndpoints {
    /// `{denoiser}.{latent_port}` — the loop-carried sample input.
    latent: String,
    /// Prompt-encoder outputs routed into denoiser conditioning ports.
    conditioning: Vec<ConditioningEdge>,
    /// `{denoiser}.{latent_port}.noise` — per-step noise for stochastic schedulers.
    noise: Option<String>,
    /// Prompt-phase encoder component name.
    text_encoder: String,
    /// Final-phase component emitting the image, when the pipeline declares one.
    final_component: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ConditioningEdge {
    encoder_output: String,
    denoiser_input: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConditioningKind {
    Single,
    DualWithPooled,
}

/// Split a `component.port` endpoint. Endpoints always carry both halves.
fn split_endpoint(endpoint: &str) -> Option<(&str, &str)> {
    endpoint.split_once('.')
}

fn resolve_endpoints(spec: &PipelineSpec) -> Result<DiffusionEndpoints> {
    let denoiser = spec.strategy.denoiser.as_deref().context(
        "What: this package cannot be rendered as text-to-image. \
         Why: its pipeline strategy declares no `denoiser` component, so there is no denoise loop to drive. \
         How: point the command at a diffusion package whose metadata declares `strategy.kind: iterative` with a `denoiser`.",
    )?;

    // The loop-carried self-edge (`{denoiser}.noise_pred -> {denoiser}.sample`)
    // names the latent input port without assuming any conventional name.
    let latent_port = spec
        .dataflow
        .iter()
        .find_map(|edge| {
            let (from_component, _) = split_endpoint(&edge.from)?;
            let (to_component, to_port) = split_endpoint(&edge.to)?;
            (from_component == denoiser && to_component == denoiser).then_some(to_port)
        })
        .with_context(|| {
            format!(
                "What: the denoise loop's latent input could not be resolved. \
                 Why: pipeline.dataflow declares no loop-carried self-edge on component '{denoiser}'. \
                 How: declare an edge from the denoiser's noise-prediction output back to its sample input."
            )
        })?;

    let cfg_port = spec.strategy.cfg_conditioning_input.as_deref();
    // The conditioning edge into the denoiser identifies the prompt encoder.
    let conditioning_target = cfg_port.map(|port| match split_endpoint(port) {
        Some((component, port)) if component == denoiser => format!("{denoiser}.{port}"),
        Some(_) => port.to_string(),
        None => format!("{denoiser}.{port}"),
    });
    let text_encoder = conditioning_target
        .as_deref()
        .and_then(|target| {
            spec.dataflow.iter().find_map(|edge| {
                let (from_component, _) = split_endpoint(&edge.from)?;
                (edge.to == target && from_component != denoiser)
                    .then(|| from_component.to_string())
            })
        })
        .or_else(|| {
            // No CFG conditioning declared: fall back to the single component
            // gated to the prompt phase.
            let prompt_phase: Vec<&String> = spec
                .phases
                .iter()
                .filter(|(_, phase)| phase.run_on == PhaseRunOn::PromptOnly)
                .map(|(name, _)| name)
                .collect();
            match prompt_phase.as_slice() {
                [only] => Some((*only).clone()),
                _ => None,
            }
        })
        .context(
            "What: the prompt encoder component could not be resolved. \
             Why: no dataflow edge feeds the denoiser's conditioning input and no single `run_on: prompt_only` component is declared. \
             How: declare `strategy.cfg_conditioning_input` plus the encoder→denoiser edge, or mark exactly one component `run_on: prompt_only`.",
        )?;
    let conditioning: Vec<ConditioningEdge> = spec
        .dataflow
        .iter()
        .filter_map(|edge| {
            let (from_component, encoder_output) = split_endpoint(&edge.from)?;
            let (to_component, denoiser_input) = split_endpoint(&edge.to)?;
            (from_component == text_encoder && to_component == denoiser).then(|| ConditioningEdge {
                encoder_output: encoder_output.to_string(),
                denoiser_input: denoiser_input.to_string(),
            })
        })
        .collect();
    if conditioning.is_empty() {
        bail!(
            "What: the prompt encoder has no declared denoiser conditioning outputs. \
             Why: no dataflow edge connects '{text_encoder}' to '{denoiser}'. \
             How: declare each encoder output → denoiser input edge in pipeline.dataflow."
        );
    }

    let final_component = spec
        .phases
        .iter()
        .find(|(_, phase)| phase.run_on == PhaseRunOn::FinalOnly)
        .map(|(name, _)| name.clone());

    let scheduler_kind = spec
        .strategy
        .scheduler_config
        .as_ref()
        .map(|scheduler| scheduler.kind.as_str())
        .or(spec.strategy.scheduler.as_deref())
        .unwrap_or_default();
    let noise = scheduler_kind
        .contains("ancestral")
        .then(|| format!("{denoiser}.{latent_port}.noise"));

    Ok(DiffusionEndpoints {
        latent: format!("{denoiser}.{latent_port}"),
        conditioning,
        noise,
        text_encoder,
        final_component,
    })
}

fn conditioning_kind(edges: &[ConditioningEdge]) -> ConditioningKind {
    if edges
        .iter()
        .any(|edge| edge.denoiser_input == "text_embeds")
    {
        ConditioningKind::DualWithPooled
    } else {
        ConditioningKind::Single
    }
}

/// Build SDXL micro-conditioning values in diffusers order:
/// `[original_height, original_width, crop_top, crop_left, target_height, target_width]`.
pub fn build_time_ids(
    batch_size: usize,
    original_size: (usize, usize),
    crop_top_left: (usize, usize),
    target_size: (usize, usize),
) -> Vec<f32> {
    let row = [
        original_size.0 as f32,
        original_size.1 as f32,
        crop_top_left.0 as f32,
        crop_top_left.1 as f32,
        target_size.0 as f32,
        target_size.1 as f32,
    ];
    row.repeat(batch_size)
}

/// Concatenate two `[batch, sequence, hidden]` tensors along the hidden axis.
pub fn concatenate_hidden_states(
    left: &[f32],
    right: &[f32],
    batch_size: usize,
    sequence_length: usize,
) -> Result<Vec<f32>> {
    let rows = batch_size
        .checked_mul(sequence_length)
        .context("hidden-state row count overflow")?;
    if rows == 0 || !left.len().is_multiple_of(rows) || !right.len().is_multiple_of(rows) {
        bail!(
            "hidden states must both have shape [batch, sequence, hidden]; got {} and {} values for batch={batch_size}, sequence={sequence_length}",
            left.len(),
            right.len()
        );
    }
    let left_hidden = left.len() / rows;
    let right_hidden = right.len() / rows;
    let mut concatenated = Vec::with_capacity(left.len() + right.len());
    for row in 0..rows {
        concatenated.extend_from_slice(&left[row * left_hidden..(row + 1) * left_hidden]);
        concatenated.extend_from_slice(&right[row * right_hidden..(row + 1) * right_hidden]);
    }
    Ok(concatenated)
}

/// Load a CLIP tokenizer configured for fixed-length ([`CLIP_CONTEXT_LENGTH`])
/// padding + truncation, so its ids match the diffusers
/// `padding="max_length", truncation=True` path.
pub fn load_clip_tokenizer(path: &Path) -> Result<tokenizers::Tokenizer> {
    let mut tokenizer = tokenizers::Tokenizer::from_file(path).map_err(|error| {
        anyhow::anyhow!(
            "What: the CLIP tokenizer could not be loaded from {}. \
             Why: {error}. \
             How: point --tokenizer at the package's CLIP tokenizer.json.",
            path.display()
        )
    })?;
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::Fixed(CLIP_CONTEXT_LENGTH),
        direction: PaddingDirection::Right,
        pad_to_multiple_of: None,
        pad_id: CLIP_END_OF_TEXT_ID,
        pad_type_id: 0,
        pad_token: "<|endoftext|>".to_string(),
    }));
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: CLIP_CONTEXT_LENGTH,
            strategy: TruncationStrategy::LongestFirst,
            direction: TruncationDirection::Right,
            stride: 0,
        }))
        .map_err(|error| anyhow::anyhow!("configuring CLIP truncation: {error}"))?;
    Ok(tokenizer)
}

/// Tokenize `text` to exactly [`CLIP_CONTEXT_LENGTH`] `i64` ids.
pub fn tokenize_clip(tokenizer: &tokenizers::Tokenizer, text: &str) -> Result<Vec<i64>> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|error| anyhow::anyhow!("tokenizing {text:?}: {error}"))?;
    Ok(encoding.get_ids().iter().map(|&id| id as i64).collect())
}

/// Tile a single-row `[len]` id vector into `[batch_size, len]`, row-major.
pub fn tile_ids(ids: &[i64], batch_size: usize) -> Vec<i64> {
    let mut tiled = Vec::with_capacity(ids.len() * batch_size);
    for _ in 0..batch_size {
        tiled.extend_from_slice(ids);
    }
    tiled
}

/// Build a float tensor `Value` from f32 data matching a model input's declared
/// float dtype (fp16 packages need fp16 inputs). Falls back to f32 for any
/// non-float target dtype.
pub fn float_input(data: &[f32], shape: &[i64], dtype: DataType) -> Result<Value> {
    match dtype {
        DataType::Float16 | DataType::BFloat16 => {
            Value::from_f32_slice_as(data, shape, dtype).map_err(Into::into)
        }
        _ => Value::from_slice_f32(data, shape).map_err(Into::into),
    }
}

/// Run a text encoder once on `input_ids` and return its hidden states.
pub fn text_encode(
    environment: &Environment,
    text_encoder_path: &Path,
    input_ids: &[i64],
    batch_size: usize,
) -> Result<Vec<f32>> {
    let _span = onnx_genai_ort::prof_span!("diffusion.text_encode");
    let session = Session::new(environment, text_encoder_path, SessionOptions::default())
        .with_context(|| {
            format!(
                "What: the prompt encoder could not be loaded from {}. \
                 Why: the ONNX session failed to initialize. \
                 How: verify the file exists in the package and is a valid ONNX model.",
                text_encoder_path.display()
            )
        })?;
    let input_name = session
        .input_names()
        .first()
        .context("the prompt encoder graph declares no inputs")?
        .clone();
    let ids_value =
        Value::from_slice_i64(input_ids, &[batch_size as i64, CLIP_CONTEXT_LENGTH as i64])?;
    let outputs = session
        .run(&[(input_name.as_str(), &ids_value)])
        .with_context(|| {
            format!(
                "What: the prompt could not be encoded for a batch of {batch_size}. \
                 Why: {} rejected an input of shape [{batch_size}, {CLIP_CONTEXT_LENGTH}]. \
                 How: a package exported with a fixed batch of 1 must be rendered one image at a time.",
                text_encoder_path.display()
            )
        })?;
    let hidden = outputs
        .into_iter()
        .next()
        .context("the prompt encoder produced no output")?;
    Ok(hidden.to_vec_f32_lossy()?)
}

fn tokenizer_for_encoder_input(
    pipeline_dir: &Path,
    requested: Option<&Path>,
    declared: Option<&str>,
    input_index: usize,
) -> Result<tokenizers::Tokenizer> {
    let primary = requested
        .map(Path::to_path_buf)
        .or_else(|| declared.map(|path| pipeline_dir.join(path)))
        .unwrap_or_else(|| pipeline_dir.join("tokenizer.json"));
    let path = if input_index == 0 {
        primary
    } else {
        let indexed = pipeline_dir.join(format!("tokenizer_{}.json", input_index + 1));
        if indexed.exists() { indexed } else { primary }
    };
    load_clip_tokenizer(&path)
}

fn encode_named(
    environment: &Environment,
    encoder_path: &Path,
    inputs: &[(String, Value)],
) -> Result<HashMap<String, Value>> {
    let session = Session::new(environment, encoder_path, SessionOptions::default())
        .with_context(|| format!("loading prompt encoder {}", encoder_path.display()))?;
    let refs: Vec<(&str, &Value)> = inputs
        .iter()
        .map(|(name, value)| (name.as_str(), value))
        .collect();
    Ok(session
        .output_names()
        .iter()
        .cloned()
        .zip(session.run(&refs)?)
        .collect())
}

fn vae_encode(
    environment: &Environment,
    encoder: &VaeEncoder,
    pixels: &[f32],
    batch_size: usize,
    height: usize,
    width: usize,
    latent_channels: usize,
) -> Result<Vec<f32>> {
    let session = Session::new(environment, &encoder.model_path, SessionOptions::default())
        .with_context(|| format!("loading VAE encoder {}", encoder.model_path.display()))?;
    let input = session
        .inputs()
        .first()
        .context("the VAE encoder graph declares no inputs")?;
    let value = float_input(
        pixels,
        &[batch_size as i64, 3, height as i64, width as i64],
        input.dtype,
    )?;
    let outputs = session.run(&[(input.name.as_str(), &value)])?;
    let encoded = outputs
        .into_iter()
        .next()
        .context("the VAE encoder produced no output")?;
    let shape = encoded.shape();
    if shape.len() != 4 {
        bail!("VAE encoder output must be rank 4, got {shape:?}");
    }
    let output_channels = shape[1] as usize;
    if output_channels != latent_channels && output_channels != latent_channels * 2 {
        bail!(
            "VAE encoder produced {output_channels} channels; expected {latent_channels} latents or {} moments",
            latent_channels * 2
        );
    }
    let encoded = encoded.to_vec_f32_lossy()?;
    let output_plane = shape[2] as usize * shape[3] as usize;
    let mut latents = Vec::with_capacity(batch_size * latent_channels * output_plane);
    for batch in 0..batch_size {
        let start = batch * output_channels * output_plane;
        latents.extend(
            encoded[start..start + latent_channels * output_plane]
                .iter()
                .map(|value| value * encoder.scaling_factor),
        );
    }
    Ok(latents)
}

fn downsample_mask(mask: &[f32], width: usize, height: usize) -> Vec<f32> {
    let latent_width = width / VAE_DOWNSCALE;
    let latent_height = height / VAE_DOWNSCALE;
    let mut downsampled = Vec::with_capacity(latent_width * latent_height);
    for y in 0..latent_height {
        for x in 0..latent_width {
            downsampled.push(mask[y * VAE_DOWNSCALE * width + x * VAE_DOWNSCALE]);
        }
    }
    downsampled
}

/// Build inpainting conditioning in UNet channel order:
/// `[1-channel mask | 4-channel masked-image latent]`.
pub fn build_inpaint_conditioning(
    mask: &[f32],
    masked_latent: &[f32],
    batch_size: usize,
    latent_channels: usize,
    latent_height: usize,
    latent_width: usize,
) -> Result<Vec<f32>> {
    let plane = latent_height * latent_width;
    if mask.len() != batch_size * plane
        || masked_latent.len() != batch_size * latent_channels * plane
    {
        bail!("inpainting mask/latent shapes do not match");
    }
    let mut conditioning = Vec::with_capacity(batch_size * (latent_channels + 1) * plane);
    for batch in 0..batch_size {
        conditioning.extend_from_slice(&mask[batch * plane..(batch + 1) * plane]);
        conditioning.extend_from_slice(
            &masked_latent[batch * latent_channels * plane..(batch + 1) * latent_channels * plane],
        );
    }
    Ok(conditioning)
}

/// Decode a `[batch, channels, h, w]` latent (already scaled by
/// `1 / scaling_factor`) through a standalone VAE decoder session.
pub fn vae_decode(
    environment: &Environment,
    vae_decoder_path: &Path,
    latent: &[f32],
    shape: &[i64],
) -> Result<(Vec<f32>, usize, usize)> {
    let _span = onnx_genai_ort::prof_span!("diffusion.vae_decode");
    let session = Session::new(environment, vae_decoder_path, SessionOptions::default())
        .with_context(|| {
            format!(
                "What: the VAE decoder could not be loaded from {}. \
                 Why: the ONNX session failed to initialize. \
                 How: pass --vae-decoder pointing at the package's latent→image ONNX file.",
                vae_decoder_path.display()
            )
        })?;
    let input_name = session
        .input_names()
        .first()
        .context("the VAE decoder graph declares no inputs")?
        .clone();
    let input_dtype = session
        .inputs()
        .first()
        .context("the VAE decoder graph declares no inputs")?
        .dtype;
    let latent_value = float_input(latent, shape, input_dtype)?;
    let outputs = session.run(&[(input_name.as_str(), &latent_value)])?;
    let image = outputs
        .into_iter()
        .next()
        .context("the VAE decoder produced no output")?;
    let image_shape = image.shape().to_vec();
    let height = image_shape[image_shape.len() - 2] as usize;
    let width = image_shape[image_shape.len() - 1] as usize;
    let image_data = image.to_vec_f32_lossy()?;
    validate_finite_decode_output(&image_data, "VAE decoder")?;
    Ok((image_data, height, width))
}

/// Convert one rendered image to an RGB8 buffer, mapping `[-1, 1]` to `[0, 255]`.
pub fn to_rgb8(image: &RenderedImage) -> Result<image::RgbImage> {
    let RenderedImage {
        width,
        height,
        pixels_chw,
    } = image;
    validate_finite_decode_output(pixels_chw, "image decoder")?;
    let plane = width * height;
    if pixels_chw.len() < plane * 3 {
        bail!(
            "What: the rendered image could not be encoded. \
             Why: {} values were produced for a {width}x{height} RGB image, which needs {}. \
             How: report this as a pipeline output-shape bug.",
            pixels_chw.len(),
            plane * 3
        );
    }
    let mut pixels = Vec::with_capacity(plane * 3);
    for y in 0..*height {
        for x in 0..*width {
            for channel in 0..3 {
                let value = pixels_chw[channel * plane + y * width + x];
                let normalized = (value / 2.0 + 0.5).clamp(0.0, 1.0);
                pixels.push((normalized * 255.0).round() as u8);
            }
        }
    }
    image::RgbImage::from_raw(*width as u32, *height as u32, pixels)
        .context("image buffer size mismatch")
}

/// Encode one rendered image as PNG bytes into `output`.
///
/// Used by callers that return images in-band (for example the server's
/// base64 image responses) rather than writing a file.
pub fn write_png(image: &RenderedImage, output: &mut Vec<u8>) -> Result<()> {
    let buffer = to_rgb8(image)?;
    buffer
        .write_to(&mut std::io::Cursor::new(output), image::ImageFormat::Png)
        .context("What: the rendered image could not be PNG-encoded. Why: the encoder rejected the RGB8 buffer. How: report this as a renderer bug.")?;
    Ok(())
}

/// Save one image as an RGB8 PNG, mapping `[-1, 1]` to `[0, 255]`.
pub fn save_png(image: &RenderedImage, path: &Path) -> Result<()> {
    let buffer = to_rgb8(image)?;
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).ok();
    }
    buffer.save(path).with_context(|| {
        format!(
            "What: the rendered image could not be written to {}. \
             Why: the file could not be created or encoded. \
             How: choose an output path in a writable directory with a supported extension (.png).",
            path.display()
        )
    })?;
    Ok(())
}

/// Read `latent_channels` from the package's `run.json`, defaulting to
/// [`DEFAULT_LATENT_CHANNELS`] when absent.
pub fn latent_channels(pipeline_dir: &Path) -> usize {
    std::fs::read_to_string(pipeline_dir.join("run.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .and_then(|value| value.get("latent_channels").and_then(|c| c.as_u64()))
        .map(|channels| channels as usize)
        .unwrap_or(DEFAULT_LATENT_CHANNELS)
}

/// Generate images from a typed prompt-and-sampling request.
///
/// `pipeline_dir` is the package root, used to resolve the tokenizer and the
/// prompt encoder file when the request does not override them.
pub fn generate_image(
    pipeline_dir: &Path,
    engine: &mut PipelineEngine,
    request: &TextToImageRequest,
) -> Result<Vec<RenderedImage>> {
    generate_image_with_denoiser_inputs(pipeline_dir, engine, request, &[])
}

/// Generate images while supplying package-defined denoiser inputs.
pub fn generate_image_with_denoiser_inputs(
    pipeline_dir: &Path,
    engine: &mut PipelineEngine,
    request: &TextToImageRequest,
    denoiser_inputs: &[DenoiserInput],
) -> Result<Vec<RenderedImage>> {
    if !request.height.is_multiple_of(VAE_DOWNSCALE) || !request.width.is_multiple_of(VAE_DOWNSCALE)
    {
        bail!(
            "What: the requested image size was rejected. \
             Why: height ({}) and width ({}) must both be multiples of {VAE_DOWNSCALE} for the VAE's spatial downsampling. \
             How: choose sizes such as 512x512, 768x512, or 640x384.",
            request.height,
            request.width
        );
    }
    validate_batch_size(request.batch_size)?;
    validate_image_size(request.width, request.height)?;
    let batch_size = request.batch_size;
    let endpoints = resolve_endpoints(engine.spec())?;
    let guidance_scale = request
        .guidance_scale
        .or(engine.spec().strategy.guidance_scale)
        .unwrap_or(1.0);
    let uses_cfg = (guidance_scale - 1.0).abs() > f32::EPSILON;
    let num_steps = request
        .steps
        .or(engine.spec().strategy.num_steps)
        .unwrap_or(25);
    validate_steps(num_steps)?;
    let start_step = request
        .start_step
        .or(engine.spec().strategy.start_step)
        .unwrap_or(0);
    if start_step > num_steps {
        bail!("start_step ({start_step}) must be <= num_steps ({num_steps})");
    }
    if let Some(source) = &request.source_image
        && (source.width != request.width || source.height != request.height)
    {
        bail!(
            "source image is {}x{} but the request is {}x{}",
            source.width,
            source.height,
            request.width,
            request.height
        );
    }

    let encoder_component = engine
        .spec()
        .models
        .get(&endpoints.text_encoder)
        .with_context(|| format!("missing prompt encoder '{}'", endpoints.text_encoder))?;
    let text_encoder_path = request
        .text_encoder_path
        .clone()
        .unwrap_or_else(|| pipeline_dir.join(&encoder_component.filename));
    let environment = Environment::new("onnx-genai-text-to-image")?;
    let encoder_session = Session::new(&environment, &text_encoder_path, SessionOptions::default())
        .with_context(|| format!("loading prompt encoder {}", text_encoder_path.display()))?;
    let mut encoder_inputs: Vec<String> = encoder_session
        .inputs()
        .iter()
        .filter(|input| input.dtype == DataType::Int64 && input.name.contains("input_ids"))
        .map(|input| input.name.clone())
        .collect();
    if encoder_inputs.is_empty() {
        encoder_inputs = encoder_session
            .inputs()
            .iter()
            .filter(|input| input.dtype == DataType::Int64)
            .map(|input| input.name.clone())
            .take(1)
            .collect();
    }
    if encoder_inputs.is_empty() {
        bail!("the prompt encoder graph declares no int64 token-id inputs");
    }
    let mut positive_encoder_inputs = Vec::with_capacity(encoder_inputs.len());
    let mut negative_encoder_inputs = Vec::with_capacity(encoder_inputs.len());
    for (index, input_name) in encoder_inputs.iter().enumerate() {
        let tokenizer = tokenizer_for_encoder_input(
            pipeline_dir,
            request.tokenizer_path.as_deref(),
            encoder_component.tokenizer.as_deref(),
            index,
        )?;
        let positive_ids = tile_ids(&tokenize_clip(&tokenizer, &request.prompt)?, batch_size);
        let negative_ids = tile_ids(
            &tokenize_clip(&tokenizer, &request.negative_prompt)?,
            batch_size,
        );
        let shape = [batch_size as i64, CLIP_CONTEXT_LENGTH as i64];
        positive_encoder_inputs.push((
            input_name.clone(),
            Value::from_slice_i64(&positive_ids, &shape)?,
        ));
        negative_encoder_inputs.push((
            input_name.clone(),
            Value::from_slice_i64(&negative_ids, &shape)?,
        ));
    }

    let channels = latent_channels(pipeline_dir);
    let latent_height = request.height / VAE_DOWNSCALE;
    let latent_width = request.width / VAE_DOWNSCALE;
    let latent_shape = [
        batch_size as i64,
        channels as i64,
        latent_height as i64,
        latent_width as i64,
    ];

    // Draw one deterministic noise tensor. Txt2img scales it by the scheduler's
    // initial sigma; img2img asks the scheduler to noise the encoded source at
    // the selected start step.
    let init_noise_sigma = engine.diffusion_init_noise_sigma().unwrap_or(1.0);
    let mut rng = StdRng::seed_from_u64(request.seed);
    let sample_len = batch_size * channels * latent_height * latent_width;
    let noise: Vec<f32> = (0..sample_len)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect();
    let mut inpaint_conditioning = None;
    let sample = if let Some(source) = &request.source_image {
        let encoder = request
            .vae_encoder
            .as_ref()
            .context("img2img/inpainting requires a VAE encoder")?;
        let pixels = source.pixels_chw.repeat(batch_size);
        let encoded = vae_encode(
            &environment,
            encoder,
            &pixels,
            batch_size,
            request.height,
            request.width,
            channels,
        )?;
        let encoded_value = Value::from_slice_f32(&encoded, &latent_shape)?;
        let noise_value = Value::from_slice_f32(&noise, &latent_shape)?;
        let noised =
            engine.diffusion_add_noise(start_step, num_steps, &encoded_value, &noise_value)?;
        if let Some(mask) = &source.mask {
            let mut masked_pixels = source.pixels_chw.clone();
            let plane = request.height * request.width;
            for channel in 0..3 {
                for index in 0..plane {
                    masked_pixels[channel * plane + index] *= 1.0 - mask[index];
                }
            }
            let masked_latent = vae_encode(
                &environment,
                encoder,
                &masked_pixels.repeat(batch_size),
                batch_size,
                request.height,
                request.width,
                channels,
            )?;
            let mask = downsample_mask(mask, request.width, request.height).repeat(batch_size);
            inpaint_conditioning = Some(build_inpaint_conditioning(
                &mask,
                &masked_latent,
                batch_size,
                channels,
                latent_height,
                latent_width,
            )?);
        }
        noised.to_vec_f32_lossy()?
    } else {
        noise
            .iter()
            .map(|normal| normal * init_noise_sigma)
            .collect()
    };

    let mut pipeline_request =
        PipelineGenerateRequest::new(GenerateRequest::new(GeneratePrompt::TokenIds(vec![])));
    for (input_name, value) in positive_encoder_inputs {
        pipeline_request = pipeline_request
            .with_input(format!("{}.{}", endpoints.text_encoder, input_name), value);
    }
    pipeline_request = pipeline_request.with_input(
        endpoints.latent.clone(),
        Value::from_slice_f32(&sample, &latent_shape)?,
    );
    if let Some(conditioning) = inpaint_conditioning {
        pipeline_request = pipeline_request.with_input(
            format!("{}.conditioning", endpoints.latent),
            Value::from_slice_f32(
                &conditioning,
                &[
                    batch_size as i64,
                    (channels + 1) as i64,
                    latent_height as i64,
                    latent_width as i64,
                ],
            )?,
        );
    }

    // Classifier-free guidance unconditional embedding: the encoding of the
    // negative prompt (empty by default), NOT zeros.
    if uses_cfg {
        let mut negative_outputs =
            encode_named(&environment, &text_encoder_path, &negative_encoder_inputs)?;
        for edge in &endpoints.conditioning {
            let value = negative_outputs
                .remove(&edge.encoder_output)
                .with_context(|| {
                    format!(
                        "prompt encoder produced no declared conditioning output '{}'",
                        edge.encoder_output
                    )
                })?;
            pipeline_request = pipeline_request.with_input(
                format!(
                    "{}.{}.uncond",
                    engine.spec().strategy.denoiser.as_deref().unwrap(),
                    edge.denoiser_input
                ),
                value,
            );
        }
    }

    if conditioning_kind(&endpoints.conditioning) == ConditioningKind::DualWithPooled {
        let time_ids = build_time_ids(
            batch_size,
            (request.height, request.width),
            (0, 0),
            (request.height, request.width),
        );
        pipeline_request = pipeline_request.with_input(
            format!(
                "{}.time_ids",
                engine.spec().strategy.denoiser.as_deref().unwrap()
            ),
            Value::from_slice_f32(&time_ids, &[batch_size as i64, 6])?,
        );
    }

    let denoiser = engine.spec().strategy.denoiser.as_deref().unwrap();
    for input in denoiser_inputs {
        pipeline_request = pipeline_request.with_input(
            format!("{denoiser}.{}", input.name),
            Value::from_slice_f32(&input.values, &input.shape)?,
        );
    }

    // Stochastic (ancestral) schedulers consume an externally supplied,
    // reproducible per-step noise tensor.
    if let Some(noise_endpoint) = &endpoints.noise {
        let noise: Vec<f32> = (0..num_steps * sample_len)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();
        let mut shape = vec![num_steps as i64];
        shape.extend_from_slice(&latent_shape);
        pipeline_request = pipeline_request.with_input(
            noise_endpoint.clone(),
            Value::from_slice_f32(&noise, &shape)?,
        );
    }

    let outputs = engine
        .run_pipeline(
            pipeline_request.with_iterative_overrides(IterativeOverrides {
                num_steps: Some(num_steps),
                guidance_scale: Some(guidance_scale),
                start_step: request.start_step,
            }),
        )
        .with_context(|| {
            format!(
                "What: the denoise loop could not be run for a {}x{} batch of {batch_size}. \
                 Why: the pipeline rejected the supplied prompt, latent, or conditioning tensors. \
                 How: check that the package's components accept these dimensions — a package \
                 exported with a fixed batch of 1 needs --batch-size 1.",
                request.width, request.height
            )
        })?;

    // Preferred path: the pipeline declares a final image phase and emits RGB directly.
    let image_output = endpoints.final_component.as_ref().and_then(|component| {
        outputs
            .iter()
            .find(|(endpoint, _)| endpoint.starts_with(&format!("{component}.")))
    });
    if let Some((_, image_value)) = image_output {
        let shape = image_value.shape().to_vec();
        let height = shape[shape.len() - 2] as usize;
        let width = shape[shape.len() - 1] as usize;
        let data = image_value.to_vec_f32_lossy()?;
        validate_finite_decode_output(&data, "pipeline image decoder")?;
        return Ok(split_batch(data, width, height, batch_size));
    }

    // Fallback: the pipeline stops at the latent, so decode it separately.
    let Some(vae_decoder) = &request.vae_decoder else {
        let mut produced: Vec<&str> = outputs.keys().map(String::as_str).collect();
        produced.sort_unstable();
        bail!(
            "What: the pipeline produced no image. \
             Why: this package declares no `run_on: final_only` image component, so it stops at the latent; it produced [{}]. \
             How: pass --vae-decoder <latent-to-image.onnx> (with --vae-scaling-factor) to decode the final latent, or use a package whose pipeline declares a final VAE phase.",
            produced.join(", ")
        );
    };
    let latent_value = outputs.get(&endpoints.latent).with_context(|| {
        format!(
            "What: the final latent could not be read. \
             Why: the pipeline produced no '{}' output. \
             How: verify the package's loop-carried denoiser self-edge is declared correctly.",
            endpoints.latent
        )
    })?;
    let latent: Vec<f32> = latent_value
        .to_vec_f32_lossy()?
        .iter()
        .map(|&value| value / vae_decoder.scaling_factor)
        .collect();
    let (data, height, width) = vae_decode(
        &environment,
        &vae_decoder.model_path,
        &latent,
        &latent_shape,
    )?;
    Ok(split_batch(data, width, height, batch_size))
}

/// Backwards-compatible name for [`generate_image`].
pub fn render(
    pipeline_dir: &Path,
    engine: &mut PipelineEngine,
    request: &TextToImageRequest,
) -> Result<Vec<RenderedImage>> {
    generate_image(pipeline_dir, engine, request)
}

/// Resolve the prompt encoder's ONNX file from the package's declared components.
/// Split a `[batch, 3, height, width]` buffer into per-image results.
fn split_batch(
    data: Vec<f32>,
    width: usize,
    height: usize,
    batch_size: usize,
) -> Vec<RenderedImage> {
    let per_image = 3 * height * width;
    (0..batch_size)
        .filter_map(|index| {
            let start = index * per_image;
            data.get(start..start + per_image)
                .map(|slice| RenderedImage {
                    width,
                    height,
                    pixels_chw: slice.to_vec(),
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_genai_metadata::{DataflowEdge, PhaseConfig, PipelineComponentSpec, PipelineStrategy};

    fn component(filename: &str, role: &str) -> PipelineComponentSpec {
        PipelineComponentSpec {
            filename: filename.to_string(),
            role: role.to_string(),
            device_preference: None,
            tokenizer: None,
            io: None,
            ports: Default::default(),
        }
    }

    fn edge(from: &str, to: &str) -> DataflowEdge {
        DataflowEdge {
            from: from.to_string(),
            to: to.to_string(),
            dtype: None,
            device_transfer: None,
        }
    }

    fn txt2img_spec() -> PipelineSpec {
        let mut spec = PipelineSpec {
            strategy: PipelineStrategy {
                denoiser: Some("denoiser".to_string()),
                cfg_conditioning_input: Some("encoder_hidden_states".to_string()),
                guidance_scale: Some(7.5),
                num_steps: Some(20),
                ..Default::default()
            },
            ..Default::default()
        };
        spec.models.insert(
            "text_encoder".to_string(),
            component("text_encoder.onnx", "encoder"),
        );
        spec.models.insert(
            "denoiser".to_string(),
            component("denoiser.onnx", "denoiser"),
        );
        spec.models
            .insert("vae".to_string(), component("vae.onnx", "vae"));
        spec.dataflow = vec![
            edge(
                "text_encoder.last_hidden_state",
                "denoiser.encoder_hidden_states",
            ),
            edge("denoiser.noise_pred", "denoiser.sample"),
            edge("denoiser.noise_pred", "vae.latent"),
        ];
        spec.phases.insert(
            "text_encoder".to_string(),
            PhaseConfig {
                run_on: PhaseRunOn::PromptOnly,
                when_present: None,
            },
        );
        spec.phases.insert(
            "vae".to_string(),
            PhaseConfig {
                run_on: PhaseRunOn::FinalOnly,
                when_present: None,
            },
        );
        spec
    }

    #[test]
    fn resolves_endpoints_from_declared_dataflow() {
        let endpoints = resolve_endpoints(&txt2img_spec()).unwrap();

        assert_eq!(endpoints.latent, "denoiser.sample");
        assert_eq!(
            endpoints.conditioning,
            vec![ConditioningEdge {
                encoder_output: "last_hidden_state".to_string(),
                denoiser_input: "encoder_hidden_states".to_string(),
            }]
        );
        assert_eq!(endpoints.text_encoder, "text_encoder");
        assert_eq!(endpoints.final_component.as_deref(), Some("vae"));
        assert!(endpoints.noise.is_none());
    }

    #[test]
    fn ancestral_schedulers_request_per_step_noise() {
        let mut spec = txt2img_spec();
        spec.strategy.scheduler = Some("euler_ancestral".to_string());

        let endpoints = resolve_endpoints(&spec).unwrap();

        assert_eq!(endpoints.noise.as_deref(), Some("denoiser.sample.noise"));
    }

    #[test]
    fn rejects_packages_without_a_denoiser() {
        let mut spec = txt2img_spec();
        spec.strategy.denoiser = None;

        let error = resolve_endpoints(&spec).expect_err("a non-diffusion package must fail closed");

        let message = error.to_string();
        assert!(message.contains("What:"), "message: {message}");
        assert!(message.contains("How:"), "message: {message}");
    }

    #[test]
    fn fp16_decode_output_rejects_every_non_finite_class_with_actionable_error() {
        let output = Value::from_slice_f16_bits(&[0x0000, 0x7e00, 0x7c00, 0xfc00, 0x3c00], &[5])
            .expect("fp16 VAE output");
        let widened = output.to_vec_f32_lossy().expect("widen fp16 output");
        let error = validate_finite_decode_output(&widened, "test VAE")
            .expect_err("non-finite decoder output must fail closed");

        let message = error.to_string();
        assert!(message.contains("test VAE"), "message: {message}");
        assert!(message.contains("NaN: 1"), "message: {message}");
        assert!(message.contains("+Inf: 1"), "message: {message}");
        assert!(message.contains("-Inf: 1"), "message: {message}");
        assert!(message.contains("fp32 VAE decoder"), "message: {message}");
    }

    #[test]
    fn fp16_decode_output_accepts_all_finite_values_without_modification() {
        let output = Value::from_slice_f16_bits(
            &[0xfbff, 0xbc00, 0x8000, 0x0000, 0x3c00, 0x0001, 0x7bff],
            &[7],
        )
        .expect("fp16 VAE output");
        let widened = output.to_vec_f32_lossy().expect("widen fp16 output");
        let original = widened.clone();

        validate_finite_decode_output(&widened, "test VAE")
            .expect("all finite decoder values must pass through");
        assert_eq!(widened, original);
        assert!(widened.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn rgb_conversion_fails_closed_on_non_finite_pixels() {
        let image = RenderedImage {
            width: 1,
            height: 1,
            pixels_chw: vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
        };

        let error = to_rgb8(&image).expect_err("PNG conversion must reject corrupt pixels");
        assert!(
            error
                .to_string()
                .contains("image decoder produced 3 non-finite values"),
            "message: {error}"
        );
    }

    #[test]
    fn split_batch_slices_one_image_per_batch_entry() {
        let images = split_batch(vec![0.0; 2 * 3 * 4 * 5], 5, 4, 2);

        assert_eq!(images.len(), 2);
        assert_eq!(images[0].width, 5);
        assert_eq!(images[0].height, 4);
        assert_eq!(images[0].pixels_chw.len(), 3 * 4 * 5);
    }

    #[test]
    fn resolve_endpoints_falls_back_to_the_single_prompt_phase_component() {
        let mut spec = txt2img_spec();
        spec.strategy.cfg_conditioning_input = None;

        let endpoints = resolve_endpoints(&spec).unwrap();

        assert_eq!(endpoints.text_encoder, "text_encoder");
        assert_eq!(
            conditioning_kind(&endpoints.conditioning),
            ConditioningKind::Single
        );
    }

    #[test]
    fn time_ids_follow_sdxl_micro_conditioning_order_and_batch_tiling() {
        let ids = build_time_ids(2, (768, 1024), (12, 34), (640, 896));
        assert_eq!(
            ids,
            vec![
                768.0, 1024.0, 12.0, 34.0, 640.0, 896.0, 768.0, 1024.0, 12.0, 34.0, 640.0, 896.0,
            ]
        );
    }

    #[test]
    fn dual_encoder_hidden_states_concatenate_feature_rows() {
        let left = vec![1.0, 2.0, 3.0, 4.0];
        let right = vec![10.0, 11.0, 12.0, 20.0, 21.0, 22.0];
        let combined = concatenate_hidden_states(&left, &right, 1, 2).unwrap();
        assert_eq!(
            combined,
            vec![1.0, 2.0, 10.0, 11.0, 12.0, 3.0, 4.0, 20.0, 21.0, 22.0]
        );
    }

    #[test]
    fn pooled_conditioning_detects_sdxl_while_single_edge_stays_sd1() {
        let single = vec![ConditioningEdge {
            encoder_output: "last_hidden_state".to_string(),
            denoiser_input: "encoder_hidden_states".to_string(),
        }];
        let mut dual = single.clone();
        dual.push(ConditioningEdge {
            encoder_output: "text_embeds".to_string(),
            denoiser_input: "text_embeds".to_string(),
        });

        assert_eq!(conditioning_kind(&single), ConditioningKind::Single);
        assert_eq!(conditioning_kind(&dual), ConditioningKind::DualWithPooled);
    }

    #[test]
    fn inpaint_conditioning_orders_mask_before_masked_image_latent() {
        let conditioning = build_inpaint_conditioning(
            &[10.0, 11.0],
            &[20.0, 21.0, 30.0, 31.0, 40.0, 41.0, 50.0, 51.0],
            1,
            4,
            1,
            2,
        )
        .unwrap();
        assert_eq!(
            conditioning,
            vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0, 50.0, 51.0,]
        );
    }
}

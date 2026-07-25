// Copyright (c) Microsoft Corporation.
//
//! Text-to-image rendering for declarative diffusion pipelines.
//!
//! This module turns a prompt plus sampling parameters into RGB images by
//! driving a `kind: iterative` pipeline package (see [`docs/DIFFUSION.md`]):
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
//! [`docs/DIFFUSION.md`]: https://github.com/justinchuby/onnx-genai/blob/main/docs/DIFFUSION.md

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

/// Standalone VAE decoder for packages whose pipeline ends at the latent.
#[derive(Debug, Clone)]
pub struct VaeDecoder {
    /// ONNX model file implementing `latent -> image`.
    pub model_path: PathBuf,
    /// The final latent is divided by this value before decoding. Classic
    /// Stable Diffusion 1.x uses `0.18215`.
    pub scaling_factor: f32,
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
        }
    }
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
    /// `{denoiser}.{cfg_port}.uncond` — the CFG unconditional embedding, when guided.
    uncond: Option<String>,
    /// `{denoiser}.{latent_port}.noise` — per-step noise for stochastic schedulers.
    noise: Option<String>,
    /// `{text_encoder}.{input_port}` — prompt token ids.
    prompt_ids: String,
    /// Prompt-phase encoder component name.
    text_encoder: String,
    /// Final-phase component emitting the image, when the pipeline declares one.
    final_component: Option<String>,
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
    let uncond = cfg_port.map(|port| {
        let port = split_endpoint(port).map_or(port, |(_, port)| port);
        format!("{denoiser}.{port}.uncond")
    });

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
        uncond,
        noise,
        prompt_ids: format!("{text_encoder}.input_ids"),
        text_encoder,
        final_component,
    })
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

/// Decode a `[batch, channels, h, w]` latent (already scaled by
/// `1 / scaling_factor`) through a standalone VAE decoder session.
pub fn vae_decode(
    environment: &Environment,
    vae_decoder_path: &Path,
    latent: &[f32],
    shape: &[i64],
) -> Result<(Vec<f32>, usize, usize)> {
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
    Ok((image.to_vec_f32_lossy()?, height, width))
}

/// Convert one rendered image to an RGB8 buffer, mapping `[-1, 1]` to `[0, 255]`.
pub fn to_rgb8(image: &RenderedImage) -> Result<image::RgbImage> {
    let RenderedImage {
        width,
        height,
        pixels_chw,
    } = image;
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

/// Render `request` through the diffusion pipeline already loaded in `engine`.
///
/// `pipeline_dir` is the package root, used to resolve the tokenizer and the
/// prompt encoder file when the request does not override them.
pub fn render(
    pipeline_dir: &Path,
    engine: &mut PipelineEngine,
    request: &TextToImageRequest,
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
    let batch_size = request.batch_size.max(1);
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

    let tokenizer_path = request
        .tokenizer_path
        .clone()
        .unwrap_or_else(|| pipeline_dir.join("tokenizer.json"));
    let tokenizer = load_clip_tokenizer(&tokenizer_path)?;
    let positive_ids = tile_ids(&tokenize_clip(&tokenizer, &request.prompt)?, batch_size);

    let channels = latent_channels(pipeline_dir);
    let latent_height = request.height / VAE_DOWNSCALE;
    let latent_width = request.width / VAE_DOWNSCALE;
    let latent_shape = [
        batch_size as i64,
        channels as i64,
        latent_height as i64,
        latent_width as i64,
    ];

    // Seed latent: standard normal noise pre-scaled into the scheduler's sigma
    // space, queried from the engine so the renderer never duplicates sigma math.
    let init_noise_sigma = engine.diffusion_init_noise_sigma().unwrap_or(1.0);
    let mut rng = StdRng::seed_from_u64(request.seed);
    let sample_len = batch_size * channels * latent_height * latent_width;
    let sample: Vec<f32> = (0..sample_len)
        .map(|_| {
            let normal: f32 = StandardNormal.sample(&mut rng);
            normal * init_noise_sigma
        })
        .collect();

    let mut pipeline_request =
        PipelineGenerateRequest::new(GenerateRequest::new(GeneratePrompt::TokenIds(vec![])));
    pipeline_request = pipeline_request.with_input(
        endpoints.prompt_ids.clone(),
        Value::from_slice_i64(
            &positive_ids,
            &[batch_size as i64, CLIP_CONTEXT_LENGTH as i64],
        )?,
    );
    pipeline_request = pipeline_request.with_input(
        endpoints.latent.clone(),
        Value::from_slice_f32(&sample, &latent_shape)?,
    );

    // Classifier-free guidance unconditional embedding: the encoding of the
    // negative prompt (empty by default), NOT zeros.
    if uses_cfg && let Some(uncond_endpoint) = &endpoints.uncond {
        let environment = Environment::new("onnx-genai-text-to-image")?;
        let text_encoder_path = request
            .text_encoder_path
            .clone()
            .map(Ok)
            .unwrap_or_else(|| {
                encoder_path(pipeline_dir, engine.spec(), &endpoints.text_encoder)
            })?;
        let negative_ids = tile_ids(
            &tokenize_clip(&tokenizer, &request.negative_prompt)?,
            batch_size,
        );
        let uncond = text_encode(&environment, &text_encoder_path, &negative_ids, batch_size)?;
        let hidden_dim = (uncond.len() / (batch_size * CLIP_CONTEXT_LENGTH)) as i64;
        pipeline_request = pipeline_request.with_input(
            uncond_endpoint.clone(),
            Value::from_slice_f32(
                &uncond,
                &[batch_size as i64, CLIP_CONTEXT_LENGTH as i64, hidden_dim],
            )?,
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
    let environment = Environment::new("onnx-genai-text-to-image")?;
    let (data, height, width) = vae_decode(
        &environment,
        &vae_decoder.model_path,
        &latent,
        &latent_shape,
    )?;
    Ok(split_batch(data, width, height, batch_size))
}

/// Resolve the prompt encoder's ONNX file from the package's declared components.
fn encoder_path(pipeline_dir: &Path, spec: &PipelineSpec, component: &str) -> Result<PathBuf> {
    let declared = spec.models.get(component).with_context(|| {
        format!(
            "What: the prompt encoder file could not be resolved. \
             Why: pipeline.models declares no component named '{component}'. \
             How: pass --text-encoder pointing at the encoder's ONNX file."
        )
    })?;
    Ok(pipeline_dir.join(&declared.filename))
}

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
            endpoints.uncond.as_deref(),
            Some("denoiser.encoder_hidden_states.uncond")
        );
        assert_eq!(endpoints.prompt_ids, "text_encoder.input_ids");
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
        assert!(endpoints.uncond.is_none());
    }
}

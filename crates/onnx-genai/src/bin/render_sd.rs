// Copyright (c) Microsoft Corporation.
//
//! Native text-to-image driver for a classic Stable Diffusion 1.x pipeline
//! exported from Mobius's from-scratch diffusers builder.
//!
//! Unlike [`run_comfyui`], which expects a flat `denoiser.onnx` / `text_encoder.onnx`
//! / `vae.onnx` package whose VAE has the latent scaling baked in and emits a
//! `vae.image` output, this binary drives the Mobius package layout directly:
//! `text_encoder/model.onnx`, `unet/model.onnx`, and `vae_decoder/model.onnx`
//! selected through `inference_metadata.yaml`. The Mobius VAE decoder matches
//! diffusers `AutoencoderKL.decode` and therefore does NOT bake in the
//! `1 / scaling_factor` latent scaling, so this driver applies it explicitly to
//! the final latent before decoding.
//!
//! It tokenizes the positive and negative prompts with the Hugging Face
//! `tokenizers` crate (CLIP `tokenizer.json`), runs the iterative denoise loop
//! through the engine (classifier-free guidance uses the negative prompt's text
//! encoding as the unconditional embedding), scales and decodes the final latent
//! to an RGB image, and writes a PNG.
//!
//! Usage:
//!   render_sd --pipeline-dir pkg/ --prompt "an astronaut riding a horse" \
//!             --steps 25 --guidance 7.5 --seed 0 --output out.png

use anyhow::{Context, Result, bail};
use clap::Parser;
use onnx_genai::engine::{
    Engine, EngineConfig, GeneratePrompt, GenerateRequest, IterativeOverrides,
    PipelineGenerateRequest,
};
use onnx_genai::ort::{DataType, Environment, Session, SessionOptions, Value};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use std::path::{Path, PathBuf};
use tokenizers::{
    PaddingDirection, PaddingParams, PaddingStrategy, TruncationDirection, TruncationParams,
    TruncationStrategy,
};

/// CLIP context length: prompts are tokenized to exactly this many ids (fixed
/// padding + truncation), matching diffusers `CLIPTokenizer`.
const CLIP_CONTEXT_LENGTH: usize = 77;

/// CLIP end-of-text token id, used as the padding id (diffusers pads to
/// `max_length` with this token rather than id 0).
const CLIP_END_OF_TEXT_ID: u32 = 49407;

/// Number of latent channels for a classic Stable Diffusion 1.x VAE.
const LATENT_CHANNELS: usize = 4;

/// VAE spatial downsampling factor (latent side = image side / 8).
const VAE_DOWNSCALE: usize = 8;

#[derive(Parser, Debug)]
#[command(about = "Render a classic Stable Diffusion 1.x prompt through the onnx-genai pipeline.")]
struct Arguments {
    /// Exported Mobius ONNX pipeline package directory.
    #[arg(long)]
    pipeline_dir: PathBuf,

    /// Positive prompt to render.
    #[arg(long)]
    prompt: String,

    /// Negative prompt used as the classifier-free-guidance unconditional
    /// embedding (empty string reproduces diffusers' default).
    #[arg(long, default_value = "")]
    negative: String,

    /// Number of denoise steps.
    #[arg(long, default_value_t = 25)]
    steps: usize,

    /// Classifier-free-guidance scale (`1.0` disables guidance).
    #[arg(long, default_value_t = 7.5)]
    guidance: f32,

    /// Random seed for the initial latent.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Output image height in pixels (must be a multiple of 8).
    #[arg(long, default_value_t = 512)]
    height: usize,

    /// Output image width in pixels (must be a multiple of 8).
    #[arg(long, default_value_t = 512)]
    width: usize,

    /// VAE latent scaling factor; the final latent is divided by this value
    /// before decoding (classic Stable Diffusion 1.x uses `0.18215`).
    #[arg(long, default_value_t = 0.18215)]
    scaling_factor: f32,

    /// CLIP `tokenizer.json` (defaults to `<pipeline-dir>/tokenizer.json`).
    #[arg(long)]
    tokenizer: Option<PathBuf>,

    /// Output PNG path.
    #[arg(long, short, default_value = "render_sd_out.png")]
    output: PathBuf,
}

/// Load a CLIP tokenizer configured for fixed-length (77) padding + truncation,
/// so its ids match the diffusers `padding="max_length", truncation=True` path.
fn load_clip_tokenizer(path: &Path) -> Result<tokenizers::Tokenizer> {
    let mut tokenizer = tokenizers::Tokenizer::from_file(path)
        .map_err(|err| anyhow::anyhow!("loading tokenizer {}: {err}", path.display()))?;
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
        .map_err(|err| anyhow::anyhow!("configuring truncation: {err}"))?;
    Ok(tokenizer)
}

/// Tokenize `text` to exactly `CLIP_CONTEXT_LENGTH` `i64` ids.
fn tokenize(tokenizer: &tokenizers::Tokenizer, text: &str) -> Result<Vec<i64>> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|err| anyhow::anyhow!("tokenizing {text:?}: {err}"))?;
    Ok(encoding.get_ids().iter().map(|&id| id as i64).collect())
}

/// Build a float tensor `Value` from f32 data matching a model input's declared
/// float dtype (fp16 packages need fp16 inputs). Falls back to f32 for any
/// non-float target dtype.
fn float_input(data: &[f32], shape: &[i64], dtype: DataType) -> Result<Value> {
    match dtype {
        DataType::Float16 | DataType::BFloat16 => {
            Value::from_f32_slice_as(data, shape, dtype).map_err(Into::into)
        }
        _ => Value::from_slice_f32(data, shape).map_err(Into::into),
    }
}

/// Run the text encoder once on `input_ids` and return `(hidden_states, shape)`.
fn text_encode(
    environment: &Environment,
    text_encoder_path: &Path,
    input_ids: &[i64],
) -> Result<(Vec<f32>, Vec<i64>)> {
    let session = Session::new(environment, text_encoder_path, SessionOptions::default())
        .with_context(|| format!("loading {}", text_encoder_path.display()))?;
    let input_name = session
        .input_names()
        .first()
        .context("text_encoder has no inputs")?
        .clone();
    let ids_value = Value::from_slice_i64(input_ids, &[1, CLIP_CONTEXT_LENGTH as i64])?;
    let outputs = session.run(&[(input_name.as_str(), &ids_value)])?;
    let hidden = outputs
        .into_iter()
        .next()
        .context("text_encoder produced no output")?;
    let shape = hidden.shape().to_vec();
    Ok((hidden.to_vec_f32_lossy()?, shape))
}

/// Decode a `[1, LATENT_CHANNELS, h, w]` latent (already scaled by
/// `1 / scaling_factor`) through the VAE decoder, returning the `[3, H, W]`
/// image in `[-1, 1]` and its `(height, width)`.
fn vae_decode(
    environment: &Environment,
    vae_decoder_path: &Path,
    latent: &[f32],
    latent_channels: usize,
    latent_height: usize,
    latent_width: usize,
) -> Result<(Vec<f32>, usize, usize)> {
    let session = Session::new(environment, vae_decoder_path, SessionOptions::default())
        .with_context(|| format!("loading {}", vae_decoder_path.display()))?;
    let input_name = session
        .input_names()
        .first()
        .context("vae_decoder has no inputs")?
        .clone();
    let input_dtype = session
        .inputs()
        .first()
        .context("vae_decoder has no inputs")?
        .dtype;
    let latent_value = float_input(
        latent,
        &[
            1,
            latent_channels as i64,
            latent_height as i64,
            latent_width as i64,
        ],
        input_dtype,
    )?;
    let outputs = session.run(&[(input_name.as_str(), &latent_value)])?;
    let image = outputs
        .into_iter()
        .next()
        .context("vae_decoder produced no output")?;
    let shape = image.shape().to_vec();
    let height = shape[shape.len() - 2] as usize;
    let width = shape[shape.len() - 1] as usize;
    Ok((image.to_vec_f32_lossy()?, height, width))
}

/// Save a single `[3, height, width]` f32 image in `[-1, 1]` as an RGB8 PNG.
fn save_png(image_chw: &[f32], height: usize, width: usize, path: &Path) -> Result<()> {
    let mut pixels = Vec::with_capacity(height * width * 3);
    for y in 0..height {
        for x in 0..width {
            for channel in 0..3 {
                let value = image_chw[channel * height * width + y * width + x];
                let normalized = (value / 2.0 + 0.5).clamp(0.0, 1.0);
                pixels.push((normalized * 255.0).round() as u8);
            }
        }
    }
    let buffer = image::RgbImage::from_raw(width as u32, height as u32, pixels)
        .context("image buffer size mismatch")?;
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).ok();
    }
    buffer
        .save(path)
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

fn main() -> Result<()> {
    let arguments = Arguments::parse();

    if arguments.height % VAE_DOWNSCALE != 0 || arguments.width % VAE_DOWNSCALE != 0 {
        bail!(
            "height ({}) and width ({}) must be multiples of {VAE_DOWNSCALE}",
            arguments.height,
            arguments.width
        );
    }
    let latent_height = arguments.height / VAE_DOWNSCALE;
    let latent_width = arguments.width / VAE_DOWNSCALE;
    let uses_cfg = (arguments.guidance - 1.0).abs() > f32::EPSILON;

    eprintln!(
        "prompt={:?} negative={:?} {} steps, guidance {}, seed {}",
        arguments.prompt, arguments.negative, arguments.steps, arguments.guidance, arguments.seed
    );

    let tokenizer_path = arguments
        .tokenizer
        .clone()
        .unwrap_or_else(|| arguments.pipeline_dir.join("tokenizer.json"));
    let tokenizer = load_clip_tokenizer(&tokenizer_path)?;
    let positive_ids = tokenize(&tokenizer, &arguments.prompt)?;
    let negative_ids = tokenize(&tokenizer, &arguments.negative)?;

    let environment = Environment::new("render_sd")?;
    let text_encoder_path = arguments.pipeline_dir.join("text_encoder/model.onnx");
    let vae_decoder_path = arguments.pipeline_dir.join("vae_decoder/model.onnx");

    let started = std::time::Instant::now();
    let mut engine = Engine::from_pipeline_dir(&arguments.pipeline_dir, EngineConfig::default())?;
    let init_noise_sigma = engine.diffusion_init_noise_sigma().unwrap_or(1.0);
    eprintln!("init_noise_sigma = {init_noise_sigma}");

    // Seed latent: standard normal noise pre-scaled into the scheduler's sigma
    // space so the runner never duplicates the sigma math.
    let mut rng = StdRng::seed_from_u64(arguments.seed);
    let sample_len = LATENT_CHANNELS * latent_height * latent_width;
    let sample: Vec<f32> = (0..sample_len)
        .map(|_| {
            let normal: f32 = StandardNormal.sample(&mut rng);
            normal * init_noise_sigma
        })
        .collect();

    // Classifier-free guidance unconditional embedding: the encoding of the
    // negative prompt (empty by default), NOT zeros.
    let uncond = if uses_cfg {
        let (encoded, _shape) = text_encode(&environment, &text_encoder_path, &negative_ids)?;
        Some(encoded)
    } else {
        None
    };

    let mut request =
        PipelineGenerateRequest::new(GenerateRequest::new(GeneratePrompt::TokenIds(vec![])));
    request = request.with_input(
        "text_encoder.input_ids",
        Value::from_slice_i64(&positive_ids, &[1, CLIP_CONTEXT_LENGTH as i64])?,
    );
    request = request.with_input(
        "denoiser.sample",
        Value::from_slice_f32(
            &sample,
            &[
                1,
                LATENT_CHANNELS as i64,
                latent_height as i64,
                latent_width as i64,
            ],
        )?,
    );
    if let Some(uncond) = &uncond {
        let hidden_dim = (uncond.len() / CLIP_CONTEXT_LENGTH) as i64;
        request = request.with_input(
            "denoiser.encoder_hidden_states.uncond",
            Value::from_slice_f32(uncond, &[1, CLIP_CONTEXT_LENGTH as i64, hidden_dim])?,
        );
    }
    request = request.with_iterative_overrides(IterativeOverrides {
        num_steps: Some(arguments.steps),
        guidance_scale: Some(arguments.guidance),
        start_step: None,
    });

    let denoise_started = std::time::Instant::now();
    let outputs = engine.run_pipeline(request)?;
    let denoise_ms = denoise_started.elapsed().as_secs_f64() * 1000.0;

    // Final denoised latent. The Mobius VAE decoder does not bake in the latent
    // scaling, so scale by 1 / scaling_factor here before decoding.
    let latent_value = outputs
        .get("denoiser.sample")
        .context("pipeline did not produce 'denoiser.sample'")?;
    let latent: Vec<f32> = latent_value
        .to_vec_f32_lossy()?
        .iter()
        .map(|&value| value / arguments.scaling_factor)
        .collect();
    let latent_finite = latent.iter().all(|value| value.is_finite());
    let latent_min = latent.iter().cloned().fold(f32::INFINITY, f32::min);
    let latent_max = latent.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    eprintln!("[latent] finite={latent_finite} min={latent_min:.4} max={latent_max:.4}");

    let (image_data, height, width) = vae_decode(
        &environment,
        &vae_decoder_path,
        &latent,
        LATENT_CHANNELS,
        latent_height,
        latent_width,
    )?;

    let finite = image_data.iter().all(|value| value.is_finite());
    let minimum = image_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let maximum = image_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = image_data.iter().sum::<f32>() / image_data.len() as f32;
    eprintln!("[render] finite={finite} min={minimum:.4} max={maximum:.4} mean={mean:.4}");

    save_png(&image_data, height, width, &arguments.output)?;
    let total_ms = started.elapsed().as_secs_f64() * 1000.0;
    let steps_per_second = arguments.steps as f64 / (denoise_ms / 1000.0);
    eprintln!("saved: {}", arguments.output.display());

    // Machine-readable summary on stdout for callers (e.g. the demo backend).
    println!(
        "{{\"output\":{:?},\"width\":{width},\"height\":{height},\"steps\":{},\
         \"guidance\":{},\"denoise_ms\":{denoise_ms:.1},\"total_ms\":{total_ms:.1},\
         \"steps_per_second\":{steps_per_second:.3}}}",
        arguments.output.display().to_string(),
        arguments.steps,
        arguments.guidance
    );

    Ok(())
}

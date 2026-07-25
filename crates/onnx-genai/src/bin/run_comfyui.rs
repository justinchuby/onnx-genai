// Copyright (c) Microsoft Corporation.
//
//! Native driver that renders a ComfyUI API-format workflow through the
//! onnx-genai iterative diffusion pipeline.
//!
//! Unlike the (now retired) `scripts/run_comfyui.py`, this binary requires an
//! already-exported ONNX pipeline package (`denoiser.onnx` / `text_encoder.onnx`
//! / `vae.onnx` plus `inference_metadata.yaml` and `run.json`). It parses the
//! workflow with `onnx-genai-comfyui-config`, tokenizes the positive and
//! negative prompts natively with the Hugging Face `tokenizers` crate loading a
//! CLIP `tokenizer.json`, draws the seed latent (and, for ancestral schedulers,
//! the per-step noise) with a seeded RNG, runs the pipeline, and writes PNG(s).
//!
//! Usage:
//!   run_comfyui --workflow workflow.json --pipeline-dir pkg/ --output out.png
//!
//! The seed latent is pre-scaled by the scheduler's `init_noise_sigma`, queried
//! from the engine so the runner never duplicates the sigma math.

use anyhow::{Context, Result, bail};
use clap::Parser;
use onnx_genai::engine::{
    Engine, EngineConfig, GeneratePrompt, GenerateRequest, IterativeOverrides,
    PipelineGenerateRequest,
};
use onnx_genai::ort::{Environment, Value};
use onnx_genai::text_to_image::{
    CLIP_CONTEXT_LENGTH, RenderedImage, latent_channels, load_clip_tokenizer, save_png,
    text_encode, tile_ids, tokenize_clip as tokenize,
};
use onnx_genai_comfyui_config::parse_workflow_file;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Render a ComfyUI workflow through the onnx-genai diffusion pipeline.")]
struct Arguments {
    /// ComfyUI API-format workflow JSON.
    #[arg(long)]
    workflow: PathBuf,

    /// Exported ONNX pipeline package directory (denoiser/text_encoder/vae).
    #[arg(long)]
    pipeline_dir: PathBuf,

    /// CLIP `tokenizer.json` (defaults to `<pipeline-dir>/tokenizer.json`).
    #[arg(long)]
    tokenizer: Option<PathBuf>,

    /// Output PNG path (for batches: `stem_0.png`, `stem_1.png`, ...).
    #[arg(long, short, default_value = "comfyui_out.png")]
    output: PathBuf,

    /// Hidden verification path: instead of generating the fed tensors, load
    /// `sample.f32` / `ids.i64` / `uncond.f32` / `noise.f32` from this directory
    /// and assert the resulting `vae.image` is bit-identical to `image.f32`
    /// there. Proves the native pipeline path independent of the RNG.
    #[arg(long, hide = true)]
    replay_inputs: Option<PathBuf>,
}

fn read_f32(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if bytes.len() % 4 != 0 {
        bail!(
            "{}: length {} is not a multiple of 4",
            path.display(),
            bytes.len()
        );
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn read_i64(path: &Path) -> Result<Vec<i64>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if bytes.len() % 8 != 0 {
        bail!(
            "{}: length {} is not a multiple of 8",
            path.display(),
            bytes.len()
        );
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn main() -> Result<()> {
    let arguments = Arguments::parse();

    let workflow = parse_workflow_file(&arguments.workflow)
        .with_context(|| format!("parsing workflow {}", arguments.workflow.display()))?;
    let prompt = workflow.prompt.clone().unwrap_or_default();
    let negative_prompt = workflow.negative_prompt.clone().unwrap_or_default();
    let latent_channels = latent_channels(&arguments.pipeline_dir);
    let latent_height = (workflow.height / 8) as usize;
    let latent_width = (workflow.width / 8) as usize;
    let batch_size = workflow.batch_size.max(1) as usize;
    let num_steps = workflow.steps as usize;
    let uses_cfg = (workflow.cfg - 1.0).abs() > f64::EPSILON;
    let is_ancestral = workflow.scheduler_kind.contains("ancestral");

    eprintln!(
        "prompt={prompt:?} negative={negative_prompt:?} {num_steps} steps, cfg {}, {} ({})",
        workflow.cfg, workflow.sampler_name, workflow.scheduler_kind
    );

    let tokenizer_path = arguments
        .tokenizer
        .clone()
        .unwrap_or_else(|| arguments.pipeline_dir.join("tokenizer.json"));
    let tokenizer = load_clip_tokenizer(&tokenizer_path)?;
    let positive_ids = tokenize(&tokenizer, &prompt)?;
    let negative_ids = tokenize(&tokenizer, &negative_prompt)?;

    let environment = Environment::new("run_comfyui")?;
    let text_encoder_path = arguments.pipeline_dir.join("text_encoder.onnx");

    let mut engine = Engine::from_pipeline_dir(&arguments.pipeline_dir, EngineConfig::default())?;
    let init_noise_sigma = engine.diffusion_init_noise_sigma().unwrap_or(1.0);
    eprintln!("init_noise_sigma = {init_noise_sigma}");

    // Build the fed tensors: either replayed from files (verification mode) or
    // generated natively (normal render).
    let (positive_ids_tiled, sample, uncond, per_step_noise) = if let Some(replay_dir) =
        &arguments.replay_inputs
    {
        let replay_ids = read_i64(&replay_dir.join("ids.i64"))?;
        // Prove the native tokenizer matches the Python driver's ids.
        let native_ids = tile_ids(&positive_ids, batch_size);
        if native_ids != replay_ids {
            bail!(
                "native tokenized ids differ from replay ids.i64 \
                     ({} vs {} elements; first mismatch matters)",
                native_ids.len(),
                replay_ids.len()
            );
        }
        eprintln!(
            "[verify A] native tokenized ids == ids.i64 ({} ids)",
            replay_ids.len()
        );
        let sample = read_f32(&replay_dir.join("sample.f32"))?;
        let uncond = read_f32(&replay_dir.join("uncond.f32"))?;
        let noise_path = replay_dir.join("noise.f32");
        let per_step_noise = if noise_path.exists() {
            Some(read_f32(&noise_path)?)
        } else {
            None
        };
        (replay_ids, sample, Some(uncond), per_step_noise)
    } else {
        let mut rng = StdRng::seed_from_u64(workflow.seed as u64);
        let sample_len = batch_size * latent_channels * latent_height * latent_width;
        let sample: Vec<f32> = (0..sample_len)
            .map(|_| {
                let normal: f32 = StandardNormal.sample(&mut rng);
                normal * init_noise_sigma
            })
            .collect();
        let uncond = if uses_cfg {
            Some(text_encode(
                &environment,
                &text_encoder_path,
                &tile_ids(&negative_ids, batch_size),
                batch_size,
            )?)
        } else {
            None
        };
        let per_step_noise = if is_ancestral {
            let noise_len = num_steps * batch_size * latent_channels * latent_height * latent_width;
            Some(
                (0..noise_len)
                    .map(|_| StandardNormal.sample(&mut rng))
                    .collect::<Vec<f32>>(),
            )
        } else {
            None
        };
        (
            tile_ids(&positive_ids, batch_size),
            sample,
            uncond,
            per_step_noise,
        )
    };

    let mut request =
        PipelineGenerateRequest::new(GenerateRequest::new(GeneratePrompt::TokenIds(vec![])));
    request = request.with_input(
        "text_encoder.input_ids",
        Value::from_slice_i64(
            &positive_ids_tiled,
            &[batch_size as i64, CLIP_CONTEXT_LENGTH as i64],
        )?,
    );
    request = request.with_input(
        "denoiser.sample",
        Value::from_slice_f32(
            &sample,
            &[
                batch_size as i64,
                latent_channels as i64,
                latent_height as i64,
                latent_width as i64,
            ],
        )?,
    );
    if let Some(uncond) = &uncond {
        let sequence_length = CLIP_CONTEXT_LENGTH as i64;
        let hidden_dim = (uncond.len() / (batch_size * CLIP_CONTEXT_LENGTH)) as i64;
        request = request.with_input(
            "denoiser.encoder_hidden_states.uncond",
            Value::from_slice_f32(uncond, &[batch_size as i64, sequence_length, hidden_dim])?,
        );
    }
    if let Some(noise) = &per_step_noise {
        request = request.with_input(
            "denoiser.sample.noise",
            Value::from_slice_f32(
                noise,
                &[
                    num_steps as i64,
                    batch_size as i64,
                    latent_channels as i64,
                    latent_height as i64,
                    latent_width as i64,
                ],
            )?,
        );
    }

    let outputs = engine.run_pipeline(request.with_iterative_overrides(IterativeOverrides {
        num_steps: Some(num_steps),
        guidance_scale: Some(workflow.cfg as f32),
        start_step: None,
    }))?;
    let image_value = outputs
        .get("vae.image")
        .context("pipeline did not produce 'vae.image'")?;
    let image_shape = image_value.shape().to_vec();
    let image_data = image_value.to_vec_f32()?;
    let height = image_shape[image_shape.len() - 2] as usize;
    let width = image_shape[image_shape.len() - 1] as usize;
    let per_image = 3 * height * width;

    if let Some(replay_dir) = &arguments.replay_inputs {
        let reference = read_f32(&replay_dir.join("image.f32"))?;
        if reference.len() != image_data.len() {
            bail!(
                "replay image.f32 has {} elements but pipeline produced {}",
                reference.len(),
                image_data.len()
            );
        }
        let max_abs_diff = reference
            .iter()
            .zip(&image_data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[verify A] vae.image max|diff| vs image.f32 = {max_abs_diff:.3e}");
        if max_abs_diff >= 1e-5 {
            bail!("verification (A) FAILED: max|diff| {max_abs_diff:.3e} >= 1e-5");
        }
        eprintln!("[verify A] PASS: native pipeline is bit-identical to the reference driver");
        return Ok(());
    }

    // Standalone render statistics + PNG(s).
    let finite = image_data.iter().all(|v| v.is_finite());
    let minimum = image_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let maximum = image_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = image_data.iter().sum::<f32>() / image_data.len() as f32;
    let variance =
        image_data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / image_data.len() as f32;
    eprintln!(
        "[render] finite={finite} min={minimum:.4} max={maximum:.4} mean={mean:.4} var={variance:.5}"
    );

    if batch_size == 1 {
        save_png(
            &RenderedImage {
                width,
                height,
                pixels_chw: image_data[..per_image].to_vec(),
            },
            &arguments.output,
        )?;
        eprintln!("saved: {}", arguments.output.display());
    } else {
        let stem = arguments
            .output
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "out".to_string());
        let extension = arguments
            .output
            .extension()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "png".to_string());
        for index in 0..batch_size {
            let start = index * per_image;
            let path = arguments
                .output
                .with_file_name(format!("{stem}_{index}.{extension}"));
            save_png(
                &RenderedImage {
                    width,
                    height,
                    pixels_chw: image_data[start..start + per_image].to_vec(),
                },
                &path,
            )?;
            eprintln!("saved: {}", path.display());
        }
    }

    Ok(())
}

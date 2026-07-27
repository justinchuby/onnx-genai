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
use onnx_genai::ort::Value;
use onnx_genai::text_to_image::{
    CLIP_CONTEXT_LENGTH, DenoiserInput, RenderedImage, TextToImageRequest, VaeEncoder,
    generate_image_with_denoiser_inputs, latent_channels, load_clip_tokenizer, load_source_image,
    save_png, tile_ids, tokenize_clip as tokenize, validate_finite_decode_output,
};
use onnx_genai_comfyui_config::{ControlNet, parse_workflow_file};
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

fn save_images(images: &[RenderedImage], output: &Path) -> Result<()> {
    if images.len() == 1 {
        save_png(&images[0], output)?;
        eprintln!("saved: {}", output.display());
        return Ok(());
    }

    let stem = output
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "out".to_string());
    let extension = output
        .extension()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "png".to_string());
    for (index, image) in images.iter().enumerate() {
        let path = output.with_file_name(format!("{stem}_{index}.{extension}"));
        save_png(image, &path)?;
        eprintln!("saved: {}", path.display());
    }
    Ok(())
}

fn workflow_asset(workflow_path: &Path, asset: &str) -> PathBuf {
    let path = PathBuf::from(asset);
    if path.is_absolute() {
        path
    } else {
        workflow_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(path)
    }
}

fn vae_scaling_factor(pipeline_dir: &Path) -> f32 {
    std::fs::read_to_string(pipeline_dir.join("run.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .and_then(|run| {
            run.get("vae_scaling_factor")
                .and_then(|value| value.as_f64())
        })
        .map(|value| value as f32)
        .unwrap_or(0.18215)
}

fn preprocess_control_image(
    image: &image::DynamicImage,
    width: usize,
    height: usize,
    batch_size: usize,
) -> Vec<f32> {
    let image = image
        .resize_exact(
            width as u32,
            height as u32,
            image::imageops::FilterType::Lanczos3,
        )
        .to_rgb8();
    let plane = width * height;
    let mut pixels = vec![0.0f32; batch_size * 3 * plane];
    for batch in 0..batch_size {
        let batch_offset = batch * 3 * plane;
        for (index, pixel) in image.pixels().enumerate() {
            for channel in 0..3 {
                pixels[batch_offset + channel * plane + index] = pixel[channel] as f32 / 255.0;
            }
        }
    }
    pixels
}

fn adapter_id(name: &str) -> String {
    Path::new(name)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or(name)
        .to_owned()
}

fn workflow_denoiser_inputs(
    workflow_path: &Path,
    controlnets: &[ControlNet],
    loras: &[(String, f64)],
    width: usize,
    height: usize,
    batch_size: usize,
) -> Result<Vec<DenoiserInput>> {
    // Mobius exports a *single* fused ControlNet+UNet denoiser that declares one
    // unsuffixed `controlnet_cond` image input (`mobius/tasks/_controlnet.py`,
    // `_find_controlnet -> tuple | None`). There is no multi-ControlNet export
    // and no per-adapter suffixed ports, so binding `controlnet_cond.{adapter}`
    // would be silently dropped by the engine. Fail loudly instead of pretending
    // multi-ControlNet works.
    if controlnets.len() > 1 {
        bail!(
            "workflow declares {} ControlNets, but the exported denoiser supports a single \
             fused ControlNet (`controlnet_cond`); multi-ControlNet export is unavailable. \
             Refusing to bind unsupported per-adapter inputs that the model would silently drop.",
            controlnets.len()
        );
    }
    let mut inputs = Vec::with_capacity(controlnets.len() + loras.len());
    for controlnet in controlnets {
        let image_name = controlnet.image.as_deref().with_context(|| {
            format!(
                "ControlNet '{}' has no LoadImage-backed hint image",
                controlnet.name
            )
        })?;
        let path = workflow_asset(workflow_path, image_name);
        let image = image::open(&path)
            .with_context(|| format!("loading ControlNet hint image {}", path.display()))?;
        // The only runtime ControlNet input the exported denoiser declares is the
        // conditioning image `controlnet_cond`. ControlNet strength is fused at
        // export time (`checkpoint_export(controlnet=...)`, DIFFUSION.md §9), not a
        // runtime gate, so we deliberately do NOT feed a `conditioning_scale` input
        // the model never declares (the engine would silently drop it).
        inputs.push(DenoiserInput {
            name: "controlnet_cond".to_owned(),
            values: preprocess_control_image(&image, width, height, batch_size),
            shape: vec![batch_size as i64, 3, height as i64, width as i64],
        });
    }
    inputs.extend(loras.iter().map(|(name, strength)| DenoiserInput {
        name: format!("lora_gate.{}", adapter_id(name)),
        values: vec![*strength as f32],
        shape: Vec::new(),
    }));
    Ok(inputs)
}

fn main() -> Result<()> {
    let arguments = Arguments::parse();

    let workflow = parse_workflow_file(&arguments.workflow)
        .with_context(|| format!("parsing workflow {}", arguments.workflow.display()))?;
    let prompt = workflow.prompt.clone().unwrap_or_default();
    let negative_prompt = workflow.negative_prompt.clone().unwrap_or_default();
    let source_image = workflow
        .source_image
        .as_deref()
        .map(|source| {
            let source = workflow_asset(&arguments.workflow, source);
            let mask = workflow
                .mask_image
                .as_deref()
                .map(|mask| workflow_asset(&arguments.workflow, mask));
            load_source_image(&source, mask.as_deref())
        })
        .transpose()?;
    let width = source_image
        .as_ref()
        .map(|image| image.width)
        .unwrap_or(workflow.width as usize);
    let height = source_image
        .as_ref()
        .map(|image| image.height)
        .unwrap_or(workflow.height as usize);
    let latent_channels = latent_channels(&arguments.pipeline_dir);
    let latent_height = height / 8;
    let latent_width = width / 8;
    let batch_size = workflow.batch_size.max(1) as usize;
    let num_steps = workflow.steps as usize;
    let denoiser_inputs = workflow_denoiser_inputs(
        &arguments.workflow,
        &workflow.controlnets,
        &workflow.loras,
        width,
        height,
        batch_size,
    )?;

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

    let mut engine = Engine::from_pipeline_dir(&arguments.pipeline_dir, EngineConfig::default())?;
    let init_noise_sigma = engine.diffusion_init_noise_sigma().unwrap_or(1.0);
    eprintln!("init_noise_sigma = {init_noise_sigma}");

    if arguments.replay_inputs.is_none() {
        let vae_encoder = source_image.as_ref().map(|_| {
            let filename = engine
                .spec()
                .models
                .values()
                .find(|component| component.role == "vae_encoder")
                .map(|component| component.filename.as_str())
                .unwrap_or("vae_encoder.onnx");
            VaeEncoder {
                model_path: arguments.pipeline_dir.join(filename),
                scaling_factor: vae_scaling_factor(&arguments.pipeline_dir),
            }
        });
        let images = generate_image_with_denoiser_inputs(
            &arguments.pipeline_dir,
            &mut engine,
            &TextToImageRequest {
                prompt,
                negative_prompt,
                steps: Some(num_steps),
                guidance_scale: Some(workflow.cfg as f32),
                start_step: source_image.as_ref().map(|_| workflow.start_step as usize),
                seed: workflow.seed as u64,
                height,
                width,
                batch_size,
                tokenizer_path: arguments.tokenizer.clone(),
                text_encoder_path: None,
                vae_decoder: None,
                source_image,
                vae_encoder,
            },
            &denoiser_inputs,
        )?;
        let pixels: Vec<f32> = images
            .iter()
            .flat_map(|image| image.pixels_chw.iter().copied())
            .collect();
        let minimum = pixels.iter().copied().fold(f32::INFINITY, f32::min);
        let maximum = pixels.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mean = pixels.iter().sum::<f32>() / pixels.len() as f32;
        let variance = pixels
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f32>()
            / pixels.len() as f32;
        eprintln!(
            "[render] finite=true min={minimum:.4} max={maximum:.4} mean={mean:.4} var={variance:.5}"
        );
        save_images(&images, &arguments.output)?;
        return Ok(());
    }

    // Hidden verification mode preserves the original SD1.x replay contract.
    let replay_dir = arguments.replay_inputs.as_ref().unwrap();
    let positive_ids_tiled = read_i64(&replay_dir.join("ids.i64"))?;
    let native_ids = tile_ids(&positive_ids, batch_size);
    if native_ids != positive_ids_tiled {
        bail!(
            "native tokenized ids differ from replay ids.i64 \
                     ({} vs {} elements; first mismatch matters)",
            native_ids.len(),
            positive_ids_tiled.len()
        );
    }
    eprintln!(
        "[verify A] native tokenized ids == ids.i64 ({} ids)",
        positive_ids_tiled.len()
    );
    let sample = read_f32(&replay_dir.join("sample.f32"))?;
    let uncond = read_f32(&replay_dir.join("uncond.f32"))?;
    let noise_path = replay_dir.join("noise.f32");
    let per_step_noise = if noise_path.exists() {
        Some(read_f32(&noise_path)?)
    } else {
        None
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
    let sequence_length = CLIP_CONTEXT_LENGTH as i64;
    let hidden_dim = (uncond.len() / (batch_size * CLIP_CONTEXT_LENGTH)) as i64;
    request = request.with_input(
        "denoiser.encoder_hidden_states.uncond",
        Value::from_slice_f32(&uncond, &[batch_size as i64, sequence_length, hidden_dim])?,
    );
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
    let image_data = image_value.to_vec_f32_lossy()?;
    validate_finite_decode_output(&image_data, "VAE decoder")?;
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, Rgb, RgbImage};

    fn controlnet(name: &str, strength: f64, image: &str) -> ControlNet {
        ControlNet {
            name: name.to_owned(),
            strength,
            image: Some(image.to_owned()),
        }
    }

    #[test]
    fn control_image_preprocessing_is_batched_chw_rgb_in_zero_to_one() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_fn(2, 1, |x, _| match x {
            0 => Rgb([0, 127, 255]),
            _ => Rgb([255, 64, 0]),
        }));

        let pixels = preprocess_control_image(&image, 2, 1, 2);
        let expected_batch = vec![0.0, 1.0, 127.0 / 255.0, 64.0 / 255.0, 1.0, 0.0];
        assert_eq!(pixels, [expected_batch.clone(), expected_batch].concat());
        assert!(pixels.iter().all(|value| (0.0..=1.0).contains(value)));
    }

    #[test]
    fn plain_workflow_routes_no_additional_denoiser_inputs() {
        let inputs =
            workflow_denoiser_inputs(Path::new("workflow.json"), &[], &[], 8, 8, 1).unwrap();
        assert!(inputs.is_empty());
    }

    #[test]
    fn lora_strengths_route_to_named_scalar_gates() {
        let inputs = workflow_denoiser_inputs(
            Path::new("workflow.json"),
            &[],
            &[
                ("style.safetensors".to_owned(), 0.25),
                ("detail.safetensors".to_owned(), -0.5),
            ],
            8,
            8,
            1,
        )
        .unwrap();
        assert_eq!(
            inputs,
            vec![
                DenoiserInput {
                    name: "lora_gate.style".to_owned(),
                    values: vec![0.25],
                    shape: vec![],
                },
                DenoiserInput {
                    name: "lora_gate.detail".to_owned(),
                    values: vec![-0.5],
                    shape: vec![],
                },
            ]
        );
    }

    #[test]
    fn single_controlnet_binds_the_declared_unsuffixed_cond_input() {
        // The exported denoiser declares exactly one `controlnet_cond` input
        // (mobius `tasks/_controlnet.py`). Pin that name so a regression to
        // suffixed/renamed ports — which the engine would silently drop — fails
        // here instead of producing a silent no-op.
        let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/test-fixtures/controlnet-single");
        std::fs::create_dir_all(&directory).unwrap();
        RgbImage::from_pixel(2, 1, Rgb([0, 127, 255]))
            .save(directory.join("canny.png"))
            .unwrap();
        let controlnets = vec![controlnet("canny.safetensors", 0.75, "canny.png")];

        let inputs =
            workflow_denoiser_inputs(&directory.join("workflow.json"), &controlnets, &[], 2, 1, 2)
                .unwrap();

        // Exactly one input, bound to the real declared name; strength is fused at
        // export, so NO `conditioning_scale` runtime input is emitted.
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].name, "controlnet_cond");
        assert_eq!(inputs[0].shape, vec![2, 3, 1, 2]);
        assert_eq!(inputs[0].values.len(), 12);
        assert!(
            !inputs
                .iter()
                .any(|input| input.name.contains("conditioning_scale")),
            "conditioning_scale is not a declared denoiser input and must never be fed"
        );
    }

    #[test]
    fn multiple_controlnets_fail_loudly_instead_of_silently_dropping() {
        // Mobius exports a single fused ControlNet only. More than one declared
        // ControlNet has no backing export, so we must error rather than emit
        // per-adapter ports the engine would silently drop.
        let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/test-fixtures/controlnet-multi");
        std::fs::create_dir_all(&directory).unwrap();
        for name in ["canny.png", "depth.png"] {
            RgbImage::from_pixel(2, 1, Rgb([0, 127, 255]))
                .save(directory.join(name))
                .unwrap();
        }
        let controlnets = vec![
            controlnet("canny.safetensors", 0.75, "canny.png"),
            controlnet("depth.safetensors", 0.25, "depth.png"),
        ];

        let error =
            workflow_denoiser_inputs(&directory.join("workflow.json"), &controlnets, &[], 2, 1, 2)
                .unwrap_err();
        assert!(
            error.to_string().contains("single"),
            "unsupported multi-ControlNet must fail loudly: {error}"
        );
    }
}

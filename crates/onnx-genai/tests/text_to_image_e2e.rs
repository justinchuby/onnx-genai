//! End-to-end test for the text-to-image renderer that backs
//! `onnx-genai generate --output-image`.
//!
//! Uses the tiny deterministic txt2img fixture built by
//! `scripts/build_tiny_txt2img.py`, which declares the full diffusion contract:
//!
//!   * `text_encoder` (`prompt_only`): `input_ids [1, 77] -> last_hidden_state [1, 77, 8]`
//!   * `denoiser` (iterative, CFG on `encoder_hidden_states`):
//!     `noise_pred = sample * 0.5 + project(encoder_hidden_states)`, loop-carried
//!     through a `denoiser.noise_pred -> denoiser.sample` self-edge
//!   * `vae` (`final_only`): `latent [1, 4, 1, 1] -> image [1, 3, 8, 8]`
//!
//! Everything is affine and deterministic, so identical inputs render identical
//! pixels and each knob (seed, negative prompt, guidance scale) must visibly
//! change the result.

use std::path::{Path, PathBuf};

use onnx_genai::engine::{EngineConfig, PipelineEngine};
use onnx_genai::text_to_image::{self, RenderedImage, TextToImageRequest};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-txt2img")
}

fn base_request() -> TextToImageRequest {
    TextToImageRequest {
        prompt: "an astronaut riding a horse".to_string(),
        negative_prompt: "blurry low quality".to_string(),
        steps: Some(3),
        width: 8,
        height: 8,
        seed: 7,
        ..TextToImageRequest::default()
    }
}

fn render(request: &TextToImageRequest) -> RenderedImage {
    let pipeline_dir = fixture();
    let mut engine = PipelineEngine::from_dir_with_config(&pipeline_dir, EngineConfig::default())
        .expect("the tiny txt2img fixture must load as a pipeline");
    let mut images = text_to_image::generate_image(&pipeline_dir, &mut engine, request)
        .expect("the tiny txt2img fixture must render");
    assert_eq!(images.len(), 1, "one image was requested");
    images.remove(0)
}

#[test]
fn renders_a_deterministic_image_through_the_declared_pipeline() {
    let image = render(&base_request());

    assert_eq!((image.width, image.height), (8, 8));
    assert_eq!(image.pixels_chw.len(), 3 * 8 * 8);
    assert!(
        image.pixels_chw.iter().all(|value| value.is_finite()),
        "the VAE must produce finite pixels"
    );
    // The same request must be reproducible: the seed fully determines the latent.
    assert_eq!(render(&base_request()).pixels_chw, image.pixels_chw);
}

#[test]
fn the_seed_negative_prompt_and_guidance_scale_all_change_the_image() {
    let baseline = render(&base_request()).pixels_chw;

    let reseeded = render(&TextToImageRequest {
        seed: 99,
        ..base_request()
    })
    .pixels_chw;
    assert_ne!(
        reseeded, baseline,
        "the seed must change the initial latent"
    );

    let renegated = render(&TextToImageRequest {
        negative_prompt: "cat".to_string(),
        ..base_request()
    })
    .pixels_chw;
    assert_ne!(
        renegated, baseline,
        "the negative prompt is the CFG unconditional embedding and must be honored"
    );

    let unguided = render(&TextToImageRequest {
        guidance_scale: Some(1.0),
        ..base_request()
    })
    .pixels_chw;
    assert_ne!(
        unguided, baseline,
        "guidance_scale 1.0 disables CFG and must change the result"
    );
}

#[test]
fn the_prompt_changes_the_image() {
    let baseline = render(&base_request()).pixels_chw;

    let other = render(&TextToImageRequest {
        prompt: "a cat".to_string(),
        ..base_request()
    })
    .pixels_chw;

    assert_ne!(
        other, baseline,
        "the prompt must reach the denoiser through the text encoder"
    );
}

#[test]
fn rejects_image_sizes_that_are_not_a_multiple_of_the_vae_downscale() {
    let pipeline_dir = fixture();
    let mut engine = PipelineEngine::from_dir_with_config(&pipeline_dir, EngineConfig::default())
        .expect("the tiny txt2img fixture must load as a pipeline");

    let error = text_to_image::render(
        &pipeline_dir,
        &mut engine,
        &TextToImageRequest {
            width: 7,
            ..base_request()
        },
    )
    .expect_err("a non-multiple-of-8 width must fail closed");

    let message = error.to_string();
    assert!(message.contains("What:"), "message: {message}");
    assert!(message.contains("How:"), "message: {message}");
}

#[test]
fn saves_a_png_that_round_trips_as_rgb8() {
    let image = render(&base_request());
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/test-fixtures/tiny-txt2img-out.png");

    text_to_image::save_png(&image, &path).expect("the rendered image must be written");

    let decoded = image::open(&path).expect("the written file must be a readable image");
    assert_eq!((decoded.width(), decoded.height()), (8, 8));
}

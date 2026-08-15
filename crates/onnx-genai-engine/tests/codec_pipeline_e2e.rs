//! End-to-end test for the **composite** pipeline strategy over a pure
//! single-pass stage chain — the audio-to-audio (neural codec) modality from
//! DESIGN.md §20.
//!
//! Exercises `PipelineEngine::run_pipeline` on the deterministic fixture built
//! by `scripts/build_tiny_codec.py`:
//!
//!   * `encoder` (`audio_encoder`): `codes[i] = (w[2i] + w[2i+1]) / 2`
//!   * `vocoder`: `audio[2i] = audio[2i+1] = codes[i] * 2`
//!
//! wired `encoder.codes -> vocoder.codes` via the pipeline `dataflow`. The two
//! stages share one tensor pool and run in declaration order, so the closed
//! form is `audio[2i] == audio[2i+1] == w[2i] + w[2i+1]`.

use onnx_genai_engine::{
    Engine, EngineConfig, GeneratePrompt, GenerateRequest, PipelineGenerateRequest,
};
use onnx_genai_ort::Value;
use std::path::{Path, PathBuf};

fn codec_fixture() -> anyhow::Result<PathBuf> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-codec")
        .canonicalize()?)
}

fn empty_request() -> PipelineGenerateRequest {
    // A composite audio pipeline ignores the token prompt; it consumes tensors.
    PipelineGenerateRequest::new(GenerateRequest::new(GeneratePrompt::TokenIds(vec![])))
}

#[test]
fn composite_codec_pipeline_runs_encoder_then_vocoder() -> anyhow::Result<()> {
    let mut engine = Engine::from_pipeline_dir(&codec_fixture()?, EngineConfig::default())?;

    let waveform: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let request = empty_request().with_input(
        "encoder.waveform",
        Value::from_slice_f32(&waveform, &[1, 16])?,
    );

    let outputs = engine.run_pipeline(request)?;

    // Intermediate encoder codes are visible in the shared pool: mean of pairs.
    let expected_codes: Vec<f32> = waveform
        .chunks_exact(2)
        .map(|pair| (pair[0] + pair[1]) / 2.0)
        .collect();
    let codes = outputs
        .get("encoder.codes")
        .expect("encoder stage output present in the shared pool")
        .to_vec_f32()?;
    assert_eq!(codes.len(), 8);
    for (got, want) in codes.iter().zip(&expected_codes) {
        assert!((got - want).abs() < 1e-5, "codes {got} != {want}");
    }

    // Vocoder reconstructs the waveform: each code doubled into an adjacent pair,
    // so audio[2i] == audio[2i+1] == w[2i] + w[2i+1].
    let expected_audio: Vec<f32> = waveform
        .chunks_exact(2)
        .flat_map(|pair| {
            let sum = pair[0] + pair[1];
            [sum, sum]
        })
        .collect();
    let audio = outputs
        .get("vocoder.audio")
        .expect("vocoder stage output present in the shared pool")
        .to_vec_f32()?;
    assert_eq!(audio.len(), 16);
    for (got, want) in audio.iter().zip(&expected_audio) {
        assert!((got - want).abs() < 1e-5, "audio {got} != {want}");
    }

    Ok(())
}

#[test]
fn generate_rejects_composite_pipeline_with_clear_error() -> anyhow::Result<()> {
    // A pure composite produces tensors, not text: generate() must steer callers
    // to run_pipeline() instead of silently doing the wrong thing.
    let mut engine = Engine::from_pipeline_dir(&codec_fixture()?, EngineConfig::default())?;
    let err = engine
        .generate(GenerateRequest::new(GeneratePrompt::TokenIds(vec![0])))
        .expect_err("generate() must reject a non-autoregressive composite pipeline");
    assert!(
        err.to_string().contains("autoregressive"),
        "unexpected error: {err}"
    );
    Ok(())
}

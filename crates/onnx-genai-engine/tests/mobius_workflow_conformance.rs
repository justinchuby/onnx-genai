//! Cross-repository conformance for generic workflow packages emitted by Mobius.
//!
//! The seven synthetic packages are checked in under
//! `tests/fixtures/onnx_genai_workflows` from Mobius base commit
//! `3b0445aec64868e7f8ecce51fd7db3aba57455de`, with executable graph corrections
//! through `92bc47efbd0b955c3f6e7cc4c741f3cd8a494506`. Every test executes in normal
//! CI; schema validation alone is not runtime conformance.

use onnx_genai_engine::{
    Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest, PipelineGenerateRequest,
};
use onnx_genai_ort::{DataType, Value};
use std::path::{Path, PathBuf};

fn root(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/onnx_genai_workflows")
        .join(name)
}

#[test]
fn mobius_decoder_workflow_executes() -> anyhow::Result<()> {
    let root = root("decoder");
    let mut engine = Engine::from_pipeline_dir(&root, EngineConfig::default())?;
    let mut options = GenerateOptions::default();
    options.max_new_tokens = 3;
    let output = engine.run_pipeline(PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![4, 5]),
        options,
    }))?;
    assert_eq!(output["tokens"].to_vec_i64()?.len(), 3);
    Ok(())
}

#[test]
fn mobius_masked_diffusion_workflow_executes() -> anyhow::Result<()> {
    let root = root("masked");
    let mut engine = Engine::from_pipeline_dir(&root, EngineConfig::default())?;
    let mut options = GenerateOptions::default();
    options.max_new_tokens = 3;
    options.seed = Some(7);
    let request = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![0, 0]),
        options,
    })
    .with_input(
        "masked_positions",
        Value::from_raw_bytes(vec![1, 0], &[1, 2], DataType::Bool)?,
    )
    .with_input("rng_offset", Value::from_slice_i64(&[0], &[1])?);
    let output = engine.run_pipeline(request)?;
    assert_eq!(output["tokens"].shape(), [1, 2]);
    Ok(())
}

#[test]
fn mobius_codec_workflow_executes() -> anyhow::Result<()> {
    let root = root("codec");
    let mut engine = Engine::from_pipeline_dir(&root, EngineConfig::default())?;
    let request =
        PipelineGenerateRequest::new(GenerateRequest::new(GeneratePrompt::TokenIds(vec![])))
            .with_input(
                "request.waveform",
                Value::from_slice_f32(&[0.25, -0.5], &[1, 1, 2])?,
            );
    let output = engine.run_pipeline(request)?;
    assert_eq!(output["waveform"].to_vec_f32()?, [0.25, -0.5]);
    Ok(())
}

#[test]
fn mobius_vlm_workflow_executes() -> anyhow::Result<()> {
    const PNG_1X1_RGB: &[u8] = &[
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8, 2,
        0, 0, 0, 144, 119, 83, 222, 0, 0, 0, 12, 73, 68, 65, 84, 8, 215, 99, 248, 207, 192, 0, 0,
        3, 1, 1, 0, 24, 221, 141, 176, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
    ];
    let mut engine = Engine::from_pipeline_dir(&root("vlm"), EngineConfig::default())?;
    let mut options = GenerateOptions::default();
    options.max_new_tokens = 2;
    let request = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![4, 5]),
        options,
    })
    .with_input(
        "request.image",
        Value::from_raw_bytes(
            PNG_1X1_RGB.to_vec(),
            &[i64::try_from(PNG_1X1_RGB.len())?],
            DataType::Uint8,
        )?,
    );
    let output = engine.run_pipeline(request)?;
    assert_eq!(output["tokens"].to_vec_i64()?.len(), 2);
    Ok(())
}

#[test]
fn mobius_euler_diffusion_workflow_executes() -> anyhow::Result<()> {
    let mut engine = Engine::from_pipeline_dir(&root("diffusion"), EngineConfig::default())?;
    let mut options = GenerateOptions::default();
    options.max_new_tokens = 2;
    let request = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![4, 5]),
        options,
    })
    .with_input(
        "request.latent",
        Value::from_slice_f32(&[0.25; 16], &[1, 4, 2, 2])?,
    );
    let output = engine.run_pipeline(request)?;
    assert_eq!(output["image"].shape(), [1, 3, 2, 2]);
    Ok(())
}

#[test]
fn mobius_tts_workflow_executes() -> anyhow::Result<()> {
    let mut engine = Engine::from_pipeline_dir(&root("tts"), EngineConfig::default())?;
    let mut options = GenerateOptions::default();
    options.max_new_tokens = 1;
    let output = engine.run_pipeline(PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![4, 5]),
        options,
    }))?;
    assert!(!output["waveform"].to_vec_f32()?.is_empty());
    Ok(())
}

#[test]
fn mobius_speculative_rejection_workflow_executes() -> anyhow::Result<()> {
    let mut engine = Engine::from_pipeline_dir(&root("speculative"), EngineConfig::default())?;
    let mut options = GenerateOptions::default();
    options.max_new_tokens = 1;
    options.seed = Some(7);
    let mut transitions = vec![-1_i64; 32];
    transitions[1] = 0;
    let request = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![4, 5, 6, 7]),
        options,
    })
    .with_input("request.slot_ids", Value::from_slice_i64(&[0], &[1])?)
    .with_input("request.grammar_state", Value::from_slice_i64(&[0], &[1])?)
    .with_input(
        "request.grammar_transition_table",
        Value::from_slice_i64(&transitions, &[1, 32])?,
    )
    .with_input("request.draft_ms", Value::from_slice_f32(&[1.0], &[1])?)
    .with_input("request.target_ms", Value::from_slice_f32(&[2.0], &[1])?)
    .with_input(
        "request.verifier.past_key_values.0.key",
        Value::from_slice_f32(&[], &[1, 2, 0, 8])?,
    );
    let output = engine.run_pipeline(request)?;
    let tokens = output["tokens.row.0"].to_vec_i64()?;
    assert_eq!(tokens, [1]);
    Ok(())
}

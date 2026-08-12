//! Cross-repository conformance for generic workflow packages emitted by Mobius.
//!
//! Generate the synthetic `decoder`, `masked`, and `codec` package directories
//! from Mobius PR #478 commit `774448b79098d92416c1aaca94c28d2972f3454b`, then run:
//!
//! ```bash
//! MOBIUS_WORKFLOW_CONFORMANCE_DIR=/path/to/generated \
//!   cargo test -p onnx-genai-engine --test mobius_workflow_conformance \
//!   -- --ignored --nocapture
//! ```

use onnx_genai_engine::{
    Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest, PipelineGenerateRequest,
};
use onnx_genai_ort::{DataType, Value};
use std::path::PathBuf;

fn root(name: &str) -> Option<PathBuf> {
    let root = std::env::var_os("MOBIUS_WORKFLOW_CONFORMANCE_DIR")?;
    Some(PathBuf::from(root).join(name))
}

#[test]
#[ignore = "requires Mobius-generated workflow packages via MOBIUS_WORKFLOW_CONFORMANCE_DIR"]
fn mobius_decoder_workflow_executes() -> anyhow::Result<()> {
    let Some(root) = root("decoder") else {
        eprintln!("skipping: set MOBIUS_WORKFLOW_CONFORMANCE_DIR");
        return Ok(());
    };
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
#[ignore = "requires Mobius-generated workflow packages via MOBIUS_WORKFLOW_CONFORMANCE_DIR"]
fn mobius_masked_diffusion_workflow_executes() -> anyhow::Result<()> {
    let Some(root) = root("masked") else {
        eprintln!("skipping: set MOBIUS_WORKFLOW_CONFORMANCE_DIR");
        return Ok(());
    };
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
#[ignore = "requires Mobius-generated workflow packages via MOBIUS_WORKFLOW_CONFORMANCE_DIR"]
fn mobius_codec_workflow_executes() -> anyhow::Result<()> {
    let Some(root) = root("codec") else {
        eprintln!("skipping: set MOBIUS_WORKFLOW_CONFORMANCE_DIR");
        return Ok(());
    };
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

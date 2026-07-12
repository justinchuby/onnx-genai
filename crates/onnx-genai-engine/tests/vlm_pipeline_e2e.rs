use std::path::{Path, PathBuf};

use onnx_genai_engine::pipeline::PipelineGenerateRequest;
use onnx_genai_engine::{Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest};
use onnx_genai_ort::{PipelineModels, Value};

fn tiny_vlm_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("models/tiny-vlm")
}

fn tiny_pixels() -> anyhow::Result<Value> {
    Value::from_vec_f32((0..12).map(|i| i as f32 / 12.0).collect(), &[1, 3, 2, 2])
        .map_err(Into::into)
}

#[test]
#[ignore = "requires gitignored models/tiny-vlm and Batty's pipeline executor; run scripts/build_tiny_vlm.py first"]
fn tiny_vlm_pipeline_engine_emits_tokens() -> anyhow::Result<()> {
    let model_dir = tiny_vlm_dir();
    if !model_dir.is_dir() {
        eprintln!(
            "skipping tiny VLM e2e scaffold: {} is absent; run scripts/build_tiny_vlm.py",
            model_dir.display()
        );
        return Ok(());
    }

    let mut engine = Engine::from_pipeline_dir(&model_dir, EngineConfig::default())?;
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![2, 3]));
    request.options = GenerateOptions {
        max_new_tokens: 2,
        temperature: 0.0,
        stop_on_eos: false,
        ..GenerateOptions::default()
    };
    let pipeline_request =
        PipelineGenerateRequest::new(request).with_input("encoder.pixel_values", tiny_pixels()?);

    let result = engine.generate_with_pipeline_request(pipeline_request)?;

    assert!(!result.token_ids.is_empty());
    Ok(())
}

#[test]
#[ignore = "requires gitignored models/tiny-vlm; run scripts/build_tiny_vlm.py first"]
fn tiny_vlm_pipeline_sessions_emit_tokens() -> anyhow::Result<()> {
    let model_dir = tiny_vlm_dir();
    if !model_dir.is_dir() {
        eprintln!(
            "skipping tiny VLM session scaffold: {} is absent; run scripts/build_tiny_vlm.py",
            model_dir.display()
        );
        return Ok(());
    }

    let pipeline = PipelineModels::load(&model_dir)?;
    assert_eq!(pipeline.directory.spec.models.len(), 2);
    assert_eq!(
        pipeline.directory.spec.dataflow[0].from,
        "encoder.image_features"
    );
    assert_eq!(
        pipeline.directory.spec.dataflow[0].to,
        "decoder.image_features"
    );

    let pixel_values = tiny_pixels()?;
    let encoder = pipeline
        .session("encoder")
        .expect("encoder session is loaded");
    let encoder_outputs = encoder.run(&[("pixel_values", &pixel_values)])?;
    let image_features = &encoder_outputs[0];
    assert_eq!(image_features.shape(), &[1, 1, 4]);

    let input_ids = Value::from_slice_i64(&[2, 3], &[1, 2])?;
    let decoder = pipeline
        .session("decoder")
        .expect("decoder session is loaded");
    let decoder_outputs = decoder.run(&[
        ("input_ids", &input_ids),
        ("image_features", image_features),
    ])?;
    let logits = &decoder_outputs[0];
    assert_eq!(logits.shape(), &[1, 2, 8]);

    let logits = logits.to_vec_f32()?;
    let last_token_logits = &logits[8..16];
    let emitted_token = last_token_logits
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(token_id, _)| token_id)
        .expect("vocab is non-empty");
    assert_eq!(emitted_token, 4, "tiny VLM decoder should emit `cat`");
    Ok(())
}

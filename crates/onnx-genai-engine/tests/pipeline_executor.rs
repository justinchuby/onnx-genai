use onnx_genai_engine::{Engine, EngineConfig, GeneratePrompt, GenerateRequest};
use std::fs;
use std::path::{Path, PathBuf};

fn tiny_fixture() -> anyhow::Result<PathBuf> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-llm")
        .canonicalize()?)
}

fn pipeline_fixture() -> anyhow::Result<PathBuf> {
    let source = tiny_fixture()?;
    let root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/test-fixtures/pipeline-tiny-llm");
    fs::create_dir_all(&root)?;
    for file in ["model.onnx.textproto", "tokenizer.json"] {
        fs::copy(source.join(file), root.join(file))?;
    }
    fs::write(
        root.join("inference_metadata.yaml"),
        r#"
pipeline:
  models:
    decoder:
      filename: model.onnx.textproto
      type: decoder
      tokenizer: tokenizer.json
  dataflow: []
  strategy:
    kind: autoregressive
    decoder: decoder
  phases:
    decoder:
      run_on: every_step
"#,
    )?;
    Ok(root)
}

fn token_request(tokens: Vec<u32>, max_new_tokens: usize) -> GenerateRequest {
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(tokens));
    request.options.max_new_tokens = max_new_tokens;
    request.options.temperature = 0.0;
    request.options.stop_on_eos = false;
    request
}

#[test]
fn single_stage_pipeline_matches_direct_greedy_tokens() -> anyhow::Result<()> {
    let request = token_request(vec![2, 4, 3], 4);

    let mut direct = Engine::from_dir(&tiny_fixture()?, EngineConfig::default())?;
    let direct_result = direct.generate(request.clone())?;

    let mut pipeline = Engine::from_pipeline_dir(&pipeline_fixture()?, EngineConfig::default())?;
    let pipeline_result = pipeline.generate(request)?;

    assert_eq!(pipeline_result.token_ids, direct_result.token_ids);
    assert_eq!(pipeline_result.text, direct_result.text);
    assert_eq!(pipeline_result.finish_reason, direct_result.finish_reason);
    Ok(())
}

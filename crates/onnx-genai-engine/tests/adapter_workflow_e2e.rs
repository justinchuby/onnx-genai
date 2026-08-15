use std::fs;
use std::path::{Path, PathBuf};

use onnx_genai_engine::{
    AdapterActivation, AdapterSelection, Engine, EngineConfig, GeneratePrompt, GenerateRequest,
    PipelineEngine, PipelineGenerateRequest,
};
use onnx_genai_ort::Value;
use sha2::{Digest, Sha256};

fn package(metadata: &str, red: &[u8], blue: &[u8]) -> anyhow::Result<PathBuf> {
    let root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/test-fixtures/adapter-workflow");
    fs::create_dir_all(&root)?;
    fs::write(root.join("inference_metadata.yaml"), metadata)?;
    fs::write(root.join("red.json"), red)?;
    fs::write(root.join("blue.json"), blue)?;
    Ok(root)
}

fn run(
    engine: &mut PipelineEngine,
    row_ids: &[i64],
    values: &[f32],
    selection: AdapterSelection,
) -> anyhow::Result<Vec<f32>> {
    let batch = i64::try_from(row_ids.len())?;
    let request = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![]),
        options: Default::default(),
    })
    .with_input("request.row_ids", Value::from_slice_i64(row_ids, &[batch])?)
    .with_input("activations", Value::from_slice_f32(values, &[batch, 2])?)
    .with_adapters(selection);
    Ok(engine.run_pipeline(request)?["result"].to_vec_f32()?)
}

#[test]
fn heterogeneous_parameter_adapters_match_independent_rows_and_compaction() -> anyhow::Result<()> {
    let red = br#"{"targets":{"projection":{"a":[1.0,0.0],"b":[1.0,2.0]}}}"#;
    let blue = br#"{"targets":{"projection":{"a":[0.0,1.0],"b":[3.0,4.0]}}}"#;
    let metadata = format!(
        r#"
schema_version: v1
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: {{ ai.onnx: 24 }}
      adapter_abis: {{ onnx-genai.parameter-overlay: "1" }}
      capabilities: [workflow_ssa, typed_emit, parameter_adapters, heterogeneous_adapter_batching]
    inputs:
      request.row_ids:
        contract: {{ dtype: int64, rank: 1, shape: [batch] }}
        role: {{ kind: runtime, version: "1.0", role: row_ids }}
        source: {{ kind: request }}
      activations:
        contract: {{ dtype: float32, rank: 2, shape: [batch, 2] }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: activations }}
    outputs:
      result:
        contract: {{ dtype: float32, rank: 2, shape: [batch, 2] }}
        role: tensor
        stage: pre_adapter
    components:
      decoder:
        implementation: {{ kind: binding }}
        ports: {{}}
      overlay:
        implementation:
          kind: adapter
          abi: onnx-genai.parameter-overlay
          version: "1"
        ports:
          inputs:
            input: {{ dtype: float32, rank: 2, shape: [batch, 2] }}
          outputs:
            output: {{ dtype: float32, rank: 2, shape: [batch, 2] }}
        contract:
          id: onnx-genai.parameter-overlay
          version: "1"
          bindings: {{ input: input, output: output }}
          parameters: {{ action: apply, component: decoder, parameter: projection }}
    adapters:
      base_model_fingerprint: synthetic-base
      row_ids: request.row_ids
      application_capability: onnx-genai.adapters
      portable_fallback: true
      cache: {{ max_entries: 2, eviction: lru }}
      artifacts:
        red:
          identity: red
          version: "1"
          base_model_fingerprint: synthetic-base
          rank: 1
          alpha: 1.0
          dtype: float32
          weights: [{{ location: red.json, sha256: {red_sha} }}]
          targets:
            - component: decoder
              parameter: projection
              weight_key: projection
              input_features: 2
              output_features: 2
        blue:
          identity: blue
          version: "1"
          base_model_fingerprint: synthetic-base
          rank: 1
          alpha: 1.0
          dtype: float32
          weights: [{{ location: blue.json, sha256: {blue_sha} }}]
          targets:
            - component: decoder
              parameter: projection
              weight_key: projection
              input_features: 2
              output_features: 2
    steps:
      - kind: invoke
        component: overlay
        inputs: {{ input: activations }}
        outputs: {{ output: adapted }}
      - kind: emit
        value: adapted
        output: result
        mode: replace
"#,
        red_sha = format!("{:x}", Sha256::digest(red)),
        blue_sha = format!("{:x}", Sha256::digest(blue)),
    );
    let root = package(&metadata, red, blue)?;
    let mut engine = Engine::from_pipeline_dir(&root, EngineConfig::default())?;
    let selection = AdapterSelection::default()
        .with_row(10, [AdapterActivation::new("red", 1.0)])
        .with_row(
            30,
            [
                AdapterActivation::new("red", 0.5),
                AdapterActivation::new("blue", 1.0),
            ],
        );

    assert_eq!(
        run(
            &mut engine,
            &[10, 20, 30],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            selection.clone(),
        )?,
        vec![2.0, 4.0, 3.0, 4.0, 25.5, 35.0]
    );
    for (row_id, input, expected) in [
        (10, [1.0, 2.0], [2.0, 4.0]),
        (20, [3.0, 4.0], [3.0, 4.0]),
        (30, [5.0, 6.0], [25.5, 35.0]),
    ] {
        assert_eq!(
            run(&mut engine, &[row_id], &input, selection.clone())?,
            expected
        );
    }
    assert_eq!(
        run(
            &mut engine,
            &[30, 10],
            &[5.0, 6.0, 1.0, 2.0],
            selection.clone(),
        )?,
        vec![25.5, 35.0, 2.0, 4.0]
    );

    let reused = AdapterSelection::default().with_row(10, [AdapterActivation::new("blue", 1.0)]);
    assert_eq!(
        run(&mut engine, &[10], &[1.0, 2.0], reused.clone())?,
        vec![7.0, 10.0]
    );
    assert_eq!(
        run(&mut engine, &[10], &[1.0, 2.0], reused)?,
        vec![7.0, 10.0]
    );
    let diagnostic = engine.adapter_lifecycle_diagnostic();
    assert_eq!(diagnostic.loads, 2);
    assert!(diagnostic.cache_hits > 0);
    assert!(diagnostic.replayed_plans > 0);
    Ok(())
}

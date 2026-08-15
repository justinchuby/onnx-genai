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
    fs::create_dir_all(root.join("adapters/red"))?;
    fs::create_dir_all(root.join("adapters/blue"))?;
    fs::write(root.join("inference_metadata.yaml"), metadata)?;
    fs::write(root.join("adapters/red/adapter.json"), red)?;
    fs::write(root.join("adapters/blue/adapter.json"), blue)?;
    Ok(root)
}

fn run(
    engine: &mut PipelineEngine,
    slot_ids: &[i64],
    request_epochs: &[i64],
    values: &[f32],
    selection: AdapterSelection,
) -> anyhow::Result<Vec<f32>> {
    let batch = i64::try_from(slot_ids.len())?;
    let mut segments = vec![-1i64; slot_ids.len() * 2];
    let mut adapter_counts = vec![0i64; slot_ids.len()];
    let mut adapter_scales = vec![0.0f32; slot_ids.len() * 2];
    for (row, (&slot_id, &request_epoch)) in slot_ids.iter().zip(request_epochs).enumerate() {
        let identity = onnx_genai_engine::AdapterSlotIdentity {
            slot_id,
            request_epoch,
        };
        if let Some(activations) = selection.rows.get(&identity) {
            adapter_counts[row] = i64::try_from(activations.len())?;
            for (slot, activation) in activations.iter().enumerate() {
                segments[row * 2 + slot] = match activation.adapter.as_str() {
                    "red" => 0,
                    "blue" => 1,
                    other => anyhow::bail!("unknown test adapter {other}"),
                };
                adapter_scales[row * 2 + slot] = activation.scale;
            }
        }
    }
    let request = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![]),
        options: Default::default(),
    })
    .with_input(
        "request.slot_ids",
        Value::from_slice_i64(slot_ids, &[batch])?,
    )
    .with_input(
        "request.request_epochs",
        Value::from_slice_i64(request_epochs, &[batch])?,
    )
    .with_input(
        "request.adapter_segments",
        Value::from_slice_i64(&segments, &[batch, 2])?,
    )
    .with_input(
        "request.adapter_counts",
        Value::from_slice_i64(&adapter_counts, &[batch])?,
    )
    .with_input(
        "request.adapter_scales",
        Value::from_slice_f32(&adapter_scales, &[batch, 2])?,
    )
    .with_input("activations", Value::from_slice_f32(values, &[batch, 2])?);
    Ok(engine.run_pipeline(request)?["result"].to_vec_f32()?)
}

#[test]
fn heterogeneous_parameter_adapters_match_independent_rows_and_compaction() -> anyhow::Result<()> {
    let red = br#"{"targets":{"projection":{"a":[1.0,0.0],"b":[1.0,2.0]}}}"#;
    let blue = br#"{"targets":{"projection":{"a":[0.0,1.0],"b":[3.0,4.0]}}}"#;
    let metadata = format!(
        r#"
schema_version: v1
adapters:
  base_model_fingerprint: onnx-genai-targeted-base-v1:sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  target_manifest:
    targets:
      - id: projection
        component: decoder
        parameter: projection
        node_name: projection
        output_name: projection.output
        activation_dtype: float32
        input_features: 2
        output_features: 2
  selection:
    slot_ids: request.slot_ids
    request_epochs: request.request_epochs
    segments: request.adapter_segments
    adapter_counts: request.adapter_counts
    scales: request.adapter_scales
    max_adapters: 2
  application_capability: onnx-genai.adapters@1
  portable_fallback: true
  cache: {{ max_entries: 2, eviction: lru }}
  artifacts:
    red:
      index: 0
      identity: red
      version: "1"
      base_model_fingerprint: onnx-genai-targeted-base-v1:sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
      rank: 1
      alpha: 1.0
      dtype: float32
      weights:
        - {{ location: adapters/red/adapter.json, loader_capability: onnx-genai.adapters.json@1,
             sha256: {red_sha}, scale_encoding: alpha_over_rank }}
      bindings:
        - {{ target: projection, weight_key: projection }}
    blue:
      index: 1
      identity: blue
      version: "1"
      base_model_fingerprint: onnx-genai-targeted-base-v1:sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
      rank: 1
      alpha: 1.0
      dtype: float32
      weights:
        - {{ location: adapters/blue/adapter.json, loader_capability: onnx-genai.adapters.json@1,
             sha256: {blue_sha}, scale_encoding: alpha_over_rank }}
      bindings:
        - {{ target: projection, weight_key: projection }}
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: {{ ai.onnx: 24 }}
      adapter_abis: {{ onnx-genai.parameter-overlay: "1" }}
      capabilities: [workflow_ssa, typed_emit, parameter_adapters, heterogeneous_adapter_batching]
    inputs:
      request.slot_ids:
        contract: {{ dtype: int64, rank: 1, shape: [batch] }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: serving.slot_ids }}
      request.request_epochs:
        contract: {{ dtype: int64, rank: 1, shape: [batch] }}
        role: {{ kind: runtime, version: "1.0", role: request_epochs }}
        source: {{ kind: request }}
      request.adapter_segments:
        contract: {{ dtype: int64, rank: 2, shape: [batch, 2] }}
        role: {{ kind: runtime, version: "1.0", role: adapter_segments }}
        source: {{ kind: request }}
      request.adapter_counts:
        contract: {{ dtype: int64, rank: 1, shape: [batch] }}
        role: {{ kind: runtime, version: "1.0", role: adapter_counts }}
        source: {{ kind: request }}
      request.adapter_scales:
        contract: {{ dtype: float32, rank: 2, shape: [batch, 2] }}
        role: {{ kind: runtime, version: "1.0", role: adapter_scales }}
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
        .with_slot(10, 0, [AdapterActivation::new("red", 1.0)])
        .with_slot(
            30,
            0,
            [
                AdapterActivation::new("red", 0.5),
                AdapterActivation::new("blue", 1.0),
            ],
        );

    assert_eq!(
        run(
            &mut engine,
            &[10, 20, 30],
            &[0, 0, 0],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            selection.clone(),
        )?,
        vec![2.0, 4.0, 3.0, 4.0, 25.5, 35.0]
    );
    for (slot_id, input, expected) in [
        (10, [1.0, 2.0], [2.0, 4.0]),
        (20, [3.0, 4.0], [3.0, 4.0]),
        (30, [5.0, 6.0], [25.5, 35.0]),
    ] {
        assert_eq!(
            run(&mut engine, &[slot_id], &[0], &input, selection.clone())?,
            expected
        );
    }
    assert_eq!(
        run(
            &mut engine,
            &[30, 10],
            &[0, 0],
            &[5.0, 6.0, 1.0, 2.0],
            selection.clone(),
        )?,
        vec![25.5, 35.0, 2.0, 4.0]
    );

    // Reusing slot 10 at a new epoch cannot inherit its prior adapter selection.
    assert_eq!(
        run(&mut engine, &[10], &[1], &[1.0, 2.0], selection.clone(),)?,
        vec![1.0, 2.0]
    );
    let reused =
        AdapterSelection::default().with_slot(10, 1, [AdapterActivation::new("blue", 1.0)]);
    assert_eq!(
        run(&mut engine, &[10], &[1], &[1.0, 2.0], reused.clone())?,
        vec![7.0, 10.0]
    );
    assert_eq!(
        run(&mut engine, &[10], &[1], &[1.0, 2.0], reused)?,
        vec![7.0, 10.0]
    );
    let diagnostic = engine.adapter_lifecycle_diagnostic();
    assert_eq!(diagnostic.loads, 2);
    assert!(diagnostic.cache_hits > 0);
    assert!(diagnostic.replayed_plans > 0);
    Ok(())
}

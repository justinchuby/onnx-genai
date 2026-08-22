use std::fs;
use std::path::{Path, PathBuf};

use onnx_genai_engine::{
    AdapterActivation, AdapterSelection, Engine, EngineConfig, GeneratePrompt, GenerateRequest,
    PipelineGenerateRequest,
};
use onnx_genai_ort::Value;

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
    engine: &mut Engine,
    values: &[f32],
    selection: AdapterSelection,
) -> anyhow::Result<Vec<f32>> {
    // Rows are identified by their position in the batch the caller submits.
    // No slot id or epoch travels through the metadata contract.
    let rows = selection.rows.len();
    let batch = i64::try_from(rows)?;
    let mut segments = vec![-1i64; rows * 2];
    let mut adapter_counts = vec![0i64; rows];
    let mut adapter_scales = vec![0.0f32; rows * 2];
    for (row, activations) in selection.rows.iter().enumerate() {
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
    let request = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![]),
        options: Default::default(),
    })
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
    let metadata = r#"
schema_version: v1
adapters:
  target_manifest:
    targets:
      - id: projection
        component: decoder
        initializer: projection
        layer_index: 0
        node_name: projection
        output_name: projection.output
        activation_dtype: float32
        input_features: 2
        output_features: 2
        rank: 1
        alpha: 1.0
  selection:
    segments: request.adapter_segments
    adapter_counts: request.adapter_counts
    scales: request.adapter_scales
    max_adapters: 2
  application_capability: onnx-genai.adapters@1
  portable_fallback: true
  cache: { max_entries: 2, eviction: lru }
  artifacts:
    red:
      index: 0
      identity: red
      version: "1"
      rank: 1
      alpha: 1.0
      dtype: float32
      weights:
        - { location: adapters/red/adapter.json, loader_capability: onnx-genai.adapters.json@1,
             scale_encoding: alpha_over_rank }
      bindings:
        - { target: projection, weight_key: projection }
    blue:
      index: 1
      identity: blue
      version: "1"
      rank: 1
      alpha: 1.0
      dtype: float32
      weights:
        - { location: adapters/blue/adapter.json, loader_capability: onnx-genai.adapters.json@1,
             scale_encoding: alpha_over_rank }
      bindings:
        - { target: projection, weight_key: projection }
pipeline:
  workflow:
    manifest:
      adapter_abis: { onnx-genai.parameter-overlay: "1" }
      capabilities: [workflow_ssa, typed_emit, parameter_adapters, heterogeneous_adapter_batching]
    inputs:
      request.adapter_segments:
        contract: { dtype: int64, rank: 2, shape: [batch, 2], batch_layout: { kind: request_aligned, axis: 0 } }
        role: { kind: runtime, version: "1.0", role: adapter_segments }
        source: { kind: request }
      request.adapter_counts:
        contract: { dtype: int64, rank: 1, shape: [batch], batch_layout: { kind: request_aligned, axis: 0 } }
        role: { kind: runtime, version: "1.0", role: adapter_counts }
        source: { kind: request }
      request.adapter_scales:
        contract: { dtype: float32, rank: 2, shape: [batch, 2], batch_layout: { kind: request_aligned, axis: 0 } }
        role: { kind: runtime, version: "1.0", role: adapter_scales }
        source: { kind: request }
      activations:
        contract: { dtype: float32, rank: 2, shape: [batch, 2], batch_layout: { kind: request_aligned, axis: 0 } }
        role: { kind: opaque }
        source: { kind: application, name: activations }
    outputs:
      result:
        contract: { dtype: float32, rank: 2, shape: [batch, 2], batch_layout: { kind: request_aligned, axis: 0 } }
        role: tensor
        stage: pre_adapter
    components:
      decoder:
        implementation: { kind: binding }
        ports: {}
      overlay:
        implementation:
          kind: adapter
          abi: onnx-genai.parameter-overlay
          version: "1"
        row_scope: { axis: 0, stateful: false }
        ports:
          inputs:
            input: { dtype: float32, rank: 2, shape: [batch, 2], batch_layout: { kind: request_aligned, axis: 0 } }
          outputs:
            output: { dtype: float32, rank: 2, shape: [batch, 2], batch_layout: { kind: request_aligned, axis: 0 } }
        contract:
          id: onnx-genai.parameter-overlay
          version: "1"
          bindings: { input: input, output: output }
          parameters: { action: apply, component: decoder, parameter: projection }
    steps:
      - kind: invoke
        component: overlay
        inputs: { input: activations }
        outputs: { output: adapted }
      - kind: emit
        value: adapted
        output: result
        mode: replace
"#;
    let root = package(metadata, red, blue)?;
    let mut engine = Engine::from_pipeline_dir(&root, EngineConfig::default())?;
    let red_only = [AdapterActivation::new("red", 1.0)];
    let none: [AdapterActivation; 0] = [];
    let composed = [
        AdapterActivation::new("red", 0.5),
        AdapterActivation::new("blue", 1.0),
    ];

    // Heterogeneous batching: three rows, three different adapter selections,
    // associated purely by batch position.
    let selection = AdapterSelection::default()
        .with_row(red_only.clone())
        .with_row(none.clone())
        .with_row(composed.clone());
    assert_eq!(
        run(&mut engine, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], selection)?,
        vec![2.0, 4.0, 3.0, 4.0, 25.5, 35.0]
    );

    // The same rows submitted alone reproduce the same values.
    for (activations, input, expected) in [
        (red_only.to_vec(), [1.0, 2.0], [2.0, 4.0]),
        (none.to_vec(), [3.0, 4.0], [3.0, 4.0]),
        (composed.to_vec(), [5.0, 6.0], [25.5, 35.0]),
    ] {
        let selection = AdapterSelection::default().with_row(activations);
        assert_eq!(run(&mut engine, &input, selection)?, expected);
    }

    // Reordering the batch reorders the selection with it: the association is
    // positional, so nothing carries over from the previous submission.
    let reordered = AdapterSelection::default()
        .with_row(composed.clone())
        .with_row(red_only.clone());
    assert_eq!(
        run(&mut engine, &[5.0, 6.0, 1.0, 2.0], reordered)?,
        vec![25.5, 35.0, 2.0, 4.0]
    );

    // A row that reuses a batch position with a different selection gets the
    // new selection; there is no identity to inherit a stale one from.
    let blue_only = AdapterSelection::default().with_row([AdapterActivation::new("blue", 1.0)]);
    assert_eq!(
        run(
            &mut engine,
            &[1.0, 2.0],
            AdapterSelection::default().with_row(none)
        )?,
        vec![1.0, 2.0]
    );
    assert_eq!(
        run(&mut engine, &[1.0, 2.0], blue_only.clone())?,
        vec![7.0, 10.0]
    );
    assert_eq!(run(&mut engine, &[1.0, 2.0], blue_only)?, vec![7.0, 10.0]);

    let diagnostic = engine.adapter_lifecycle_diagnostic();
    assert_eq!(diagnostic.loads, 2);
    assert!(diagnostic.cache_hits > 0);
    assert!(diagnostic.replayed_plans > 0);
    Ok(())
}

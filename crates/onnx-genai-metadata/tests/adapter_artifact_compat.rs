use std::fs;
use std::path::{Path, PathBuf};

use onnx_genai_metadata::load_metadata_package;

fn package_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/test-fixtures/lora-pr-compat")
}

#[test]
fn phase1_peft_and_phase2_ort_artifacts_load_through_one_manifest() {
    let root = package_root();
    let peft_config = br#"{"r":2,"lora_alpha":4,"target_modules":["projection"]}"#;
    let safetensors = b"phase-1-peft-safetensors";
    let ort_adapter = b"TORT-phase-2-flatbuffer";
    fs::create_dir_all(root.join("adapters/peft")).expect("create PEFT fixture");
    fs::create_dir_all(root.join("adapters/ort")).expect("create ORT fixture");
    fs::write(root.join("adapters/peft/adapter_config.json"), peft_config)
        .expect("write PEFT config");
    fs::write(
        root.join("adapters/peft/adapter_model.safetensors"),
        safetensors,
    )
    .expect("write PEFT weights");
    fs::write(root.join("adapters/ort/adapter.onnx_adapter"), ort_adapter)
        .expect("write ORT adapter");

    let metadata = format!(
        r#"
schema_version: v1
adapters:
  target_manifest:
    targets:
      - id: projection
        component: decoder
        initializer: projection.weight
        layer_index: 0
        node_name: projection
        output_name: projection.output
        activation_dtype: float32
        input_features: 2
        output_features: 2
        rank: 2
        alpha: 4.0
        graph_inputs: {{ a: lora.projection.a, b: lora.projection.b }}
  discovery_fallback: tooling_only
  selection:
    segments: request.lora_segments
    adapter_counts: request.lora_counts
    scales: request.lora_scales
    max_adapters: 2
  application_capability: onnx-genai.adapters@1
  portable_fallback: true
  artifacts:
    peft:
      index: 0
      identity: phase1.peft
      version: "1"
      rank: 2
      alpha: 4.0
      dtype: float32
      weights:
        - location: adapters/peft/adapter_model.safetensors
          loader_capability: onnx-genai.adapters.hf-peft@1
          config_location: adapters/peft/adapter_config.json
          scale_encoding: alpha_over_rank
          format: hf_peft
      bindings: [{{ target: projection, weight_key: projection }}]
    ort:
      index: 1
      identity: phase2.ort
      version: "1"
      rank: 2
      alpha: 4.0
      dtype: float32
      weights:
        - location: adapters/ort/adapter.onnx_adapter
          loader_capability: onnxruntime.lora-adapter@1
          scale_encoding: baked
          format: ort_genai
      bindings: [{{ target: projection, weight_key: projection }}]
pipeline:
  workflow:
    manifest:
      capabilities: [workflow_ssa, parameter_adapters]
    inputs:
      request.lora_segments:
        contract: {{ dtype: int64, rank: 2, shape: [batch, 2],
                     batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: runtime, version: "1.0", role: adapter_segments }}
        source: {{ kind: request }}
      request.lora_counts:
        contract: {{ dtype: int64, rank: 1, shape: [batch],
                     batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: runtime, version: "1.0", role: adapter_counts }}
        source: {{ kind: request }}
      request.lora_scales:
        contract: {{ dtype: float32, rank: 2, shape: [batch, 2],
                     batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: runtime, version: "1.0", role: adapter_scales }}
        source: {{ kind: request }}
    components:
      decoder:
        implementation: {{ kind: binding }}
        ports: {{}}
    steps:
      - kind: invoke
        component: decoder
"#
    );
    fs::write(root.join("inference_metadata.yaml"), metadata).expect("write metadata");

    let loaded = load_metadata_package(&root).expect("load migrated Phase-1/Phase-2 package");
    let service = loaded.adapters.expect("adapter service");
    assert_eq!(service.artifacts.len(), 2);
    assert_eq!(service.target_manifest.targets[0].id, "projection");

    fs::write(root.join("adapters/ort/adapter.onnx_adapter"), b"corrupt")
        .expect("replace ORT adapter");
    load_metadata_package(&root)
        .expect("artifact bytes may be replaced without rewriting inference metadata");
}

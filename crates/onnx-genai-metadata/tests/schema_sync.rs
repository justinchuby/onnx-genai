use std::{fs, path::PathBuf};

use onnx_genai_metadata::inference_metadata_schema_json;

#[test]
fn committed_inference_metadata_schema_is_current() {
    let generated = format!(
        "{}\n",
        inference_metadata_schema_json().expect("schema serializes")
    );
    let path = schema_path();
    let committed = fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "failed to read {}: {error}; regenerate it with \
             `cargo run -p onnx-genai-metadata --bin gen_schema`",
            path.display()
        )
    });

    assert_eq!(
        committed,
        generated,
        "{} is out of date; regenerate it with \
         `cargo run -p onnx-genai-metadata --bin gen_schema`",
        path.display()
    );
}

#[test]
fn generated_schema_preserves_all_root_constraints() {
    let schema: serde_json::Value =
        serde_json::from_str(&inference_metadata_schema_json().expect("schema serializes"))
            .expect("generated schema is JSON");
    let constraints = schema["allOf"].as_array().expect("root allOf array");

    assert!(constraints.iter().any(|constraint| {
        constraint["not"]["required"] == serde_json::json!(["speculative", "speculator_config"])
    }));
    assert!(constraints.iter().any(|constraint| {
        constraint["not"]["required"] == serde_json::json!(["pipeline", "model"])
            && constraint["not"]["properties"]["model"]["required"] == serde_json::json!(["io"])
    }));

    let serialized = serde_json::to_string(&schema).expect("schema serializes");
    for removed in [
        "PipelineStrategy",
        "PipelineStrategyKind",
        "PhaseConfig",
        "PhaseRunOn",
        "SchedulerSpec",
        "PolicyComponentContract",
        "AdapterComponentContract",
        "ProgramOperation",
        "ControlFlow",
        "WorkflowNode",
        "WorkflowLoopCarry",
        "EffectTransition",
    ] {
        assert!(
            !serialized.contains(removed),
            "generated schema still exposes removed legacy definition {removed}"
        );
    }
    for compiler_field in [
        "initial_effects",
        "effect_name",
        "read_effect",
        "write_effect",
        "body_input",
        "body_output",
    ] {
        assert!(
            !serialized.contains(&format!("\"{compiler_field}\"")),
            "generated schema exposes compiler bookkeeping field {compiler_field}"
        );
    }
    assert!(!serialized.contains("\"kind\":{\"const\":\"transfer\""));
    assert!(serialized.contains("\"application_overridable\""));
}

fn schema_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("schema/inference_metadata.schema.json")
}

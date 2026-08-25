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

    // The published schema must forbid the retired `model.io` key outright, not
    // merely beside a `pipeline`. `ModelCapabilities` does not deny unknown
    // fields, so a producer validating against the schema alone would otherwise
    // be told its package is valid and then find the loader refuses it — the
    // schema and `parser::reject_retired_model_io` have to agree on what is
    // loadable.
    assert!(
        constraints.iter().any(|constraint| {
            constraint["not"]["required"] == serde_json::json!(["model"])
                && constraint["not"]["properties"]["model"]["required"] == serde_json::json!(["io"])
        }),
        "the schema must reject model.io wherever it appears: {constraints:#?}"
    );

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
    assert!(!serialized.contains("\"kind\":{\"const\":\"execution_island\""));
    assert!(serialized.contains("\"application_overridable\""));
    assert!(serialized.contains("\"sampling_min_p\""));
    assert!(!serialized.contains("\"custom_op_versions\""));
    assert!(!serialized.contains("\"custom_ops\""));
    assert!(schema["properties"]["adapters"].is_object());
    assert!(
        schema["$defs"]["WorkflowSpec"]["properties"]["adapters"].is_null(),
        "adapter catalog must have one top-level source of truth"
    );
    assert!(serialized.contains("\"LoraTargetManifest\""));
    assert!(serialized.contains("\"hf_peft\""));
    assert!(serialized.contains("\"segments\""));
    assert!(!serialized.contains("\"adapter_ids\""));
}

/// The published schema refuses the retired flat `token_packed` spelling, so a
/// producer is not told its package is valid and then refused by the loader.
///
/// `token_packed` used to carry `offsets` and `owner` directly; it now carries a
/// `levels` chain, and this is a reshape rather than an addition — the old
/// spelling does not load at any declared version. The two readers of a document
/// have to agree about that. `parser::reject_flat_token_packed` states it as an
/// error naming the migration; the schema states it structurally, by requiring
/// `levels` and admitting nothing else. This pins the second half, because a
/// schema that merely omitted the retired keys while allowing extras would
/// quietly bless a document the loader refuses.
#[test]
fn the_published_schema_agrees_that_the_flat_packed_spelling_is_gone() {
    let schema: serde_json::Value =
        serde_json::from_str(&inference_metadata_schema_json().expect("schema serializes"))
            .expect("generated schema is JSON");
    let packed = schema["$defs"]["BatchLayout"]["oneOf"]
        .as_array()
        .expect("batch layout variants")
        .iter()
        .find(|variant| variant["properties"]["kind"]["const"] == "token_packed")
        .expect("a token_packed variant");

    assert_eq!(
        packed["additionalProperties"],
        serde_json::json!(false),
        "the packed layout must not admit the retired keys as extras: {packed:#?}"
    );
    let required = packed["required"].as_array().expect("required list");
    assert!(
        required.contains(&serde_json::json!("levels")),
        "the packed layout must require its ownership chain: {packed:#?}"
    );
    for retired in ["offsets", "owner"] {
        assert!(
            packed["properties"][retired].is_null(),
            "the packed layout must not publish the retired `{retired}` key: {packed:#?}"
        );
    }
}

fn schema_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("schema/inference_metadata.schema.json")
}

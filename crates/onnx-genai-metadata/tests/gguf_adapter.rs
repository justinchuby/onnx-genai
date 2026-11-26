use onnx_genai_metadata::{parse_metadata_json, validate_metadata};
use serde_json::json;
fn document() -> serde_json::Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/onnx_genai_workflows/adapter/inference_metadata.yaml");
    let yaml: serde_yaml::Value =
        serde_yaml::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut doc = serde_json::to_value(yaml).unwrap();
    doc["schema_version"] = json!("1.19");
    doc["pipeline"]["workflow"]["publication_mode"] = json!("commit_only");
    doc["pipeline"]["workflow"]["outputs"]["result"]["family"] = json!({"kind":"materialized"});
    doc["adapters"]["artifacts"]["peft"]["weights"] = json!([{"format":"gguf","location":"adapters/peft/adapter.gguf","loader_capability":"onnx-genai.adapters.gguf@1","scale_encoding":"alpha_over_rank"}]);
    doc
}
#[test]
fn gguf_adapter_format_has_explicit_schema_floor_and_exact_loader_contract() {
    let doc = document();
    let metadata = parse_metadata_json(&doc).unwrap();
    validate_metadata(&metadata).unwrap();
    let mut old = doc.clone();
    old["schema_version"] = json!("1.18");
    let error = parse_metadata_json(&old).unwrap_err().to_string();
    assert!(
        error.contains("adapters.artifacts.peft.weights[0].format") && error.contains("1.19"),
        "{error}"
    );
    let mut authored = metadata;
    authored.schema_version = Some("1.18".into());
    let error = validate_metadata(&authored).unwrap_err().join("\n");
    assert!(
        error.contains("GGUF") || error.contains("gguf") || error.contains("1.19"),
        "{error}"
    );
    for (field, value, message) in [
        (
            "loader_capability",
            json!("onnx-genai.adapters.hf-peft@1"),
            "loader_capability",
        ),
        (
            "config_location",
            json!("adapters/peft/config.json"),
            "other formats forbid",
        ),
        ("scale_encoding", json!("baked"), "alpha_over_rank"),
    ] {
        let mut bad = doc.clone();
        bad["adapters"]["artifacts"]["peft"]["weights"][0][field] = value;
        let error = validate_metadata(&parse_metadata_json(&bad).unwrap())
            .unwrap_err()
            .join("\n");
        assert!(error.contains(message), "{field}: {error}");
    }
}

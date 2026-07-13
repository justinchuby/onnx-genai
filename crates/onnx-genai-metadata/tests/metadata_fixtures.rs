use std::path::{Path, PathBuf};

use onnx_genai_metadata::{
    MetadataError, PipelineStrategyKind, RuntimeCapabilities, load_metadata, load_pipeline_spec,
    validate, validate_pipeline_spec,
};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("tests/fixtures")
        .join(name)
}

fn crate_fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

#[test]
fn parses_valid_yaml_fixture() {
    let metadata =
        load_metadata(&fixture("sample_metadata.yaml")).expect("valid YAML fixture parses");

    assert_eq!(
        metadata.required_capabilities,
        ["kv_cache", "grouped_query_attention"]
    );

    let model = metadata.model.expect("model section");
    let attention = model.attention.expect("attention section");
    assert_eq!(attention.attention_type, "grouped_query");
    assert_eq!(attention.num_kv_heads, Some(8));
    assert_eq!(attention.num_attention_heads, Some(32));
    assert_eq!(attention.head_dim, Some(128));
    assert_eq!(model.max_sequence_length, Some(131_072));

    let kv_cache = metadata.kv_cache.expect("kv_cache section");
    assert_eq!(kv_cache.native_dtype.as_deref(), Some("bf16"));
    assert_eq!(
        kv_cache.sensitive_layers.as_deref(),
        Some([0, 1, -1].as_slice())
    );

    let structured_output = metadata
        .structured_output
        .expect("structured_output section");
    assert_eq!(
        structured_output.supported_formats.as_deref(),
        Some(
            [
                "json_schema".to_string(),
                "regex".to_string(),
                "context_free_grammar".to_string()
            ]
            .as_slice()
        )
    );
}

#[test]
fn parses_valid_json_fixture() {
    let metadata =
        load_metadata(&fixture("sample_metadata.json")).expect("valid JSON fixture parses");

    assert_eq!(
        metadata.required_capabilities,
        ["kv_cache", "grouped_query_attention"]
    );
    let runtime_configurable = metadata
        .model
        .expect("model section")
        .runtime_configurable
        .expect("runtime_configurable section");
    assert_eq!(runtime_configurable.prefix_cache, Some(true));
    assert_eq!(runtime_configurable.continuous_batching, Some(true));
    assert_eq!(
        runtime_configurable
            .kv_cache
            .expect("kv_cache config")
            .dtype,
        ["fp16", "fp8_e5m2"]
    );
}

#[test]
fn malformed_yaml_returns_parse_error() {
    let err = load_metadata(&fixture("malformed_metadata.yaml")).expect_err("malformed YAML fails");

    assert!(matches!(err, MetadataError::Parse(_)));
    assert!(!err.to_string().is_empty());
}

#[test]
fn schema_type_mismatch_returns_parse_error() {
    let err = load_metadata(&fixture("invalid_metadata.yaml")).expect_err("invalid schema fails");

    assert!(matches!(err, MetadataError::Parse(_)));
    assert!(err.to_string().contains("Parse error"));
}

#[test]
fn capability_validation_accepts_default_runtime_capabilities() {
    let metadata =
        load_metadata(&fixture("sample_metadata.yaml")).expect("valid YAML fixture parses");

    validate(&metadata, &RuntimeCapabilities::default()).expect("default runtime supports fixture");
}

#[test]
fn capability_validation_reports_all_unsupported_capabilities() {
    let mut metadata =
        load_metadata(&fixture("sample_metadata.yaml")).expect("valid YAML fixture parses");
    metadata
        .required_capabilities
        .push("vision_encoder".to_string());
    metadata
        .required_capabilities
        .push("speculative_decoding".to_string());

    let unsupported = validate(&metadata, &RuntimeCapabilities::default())
        .expect_err("unsupported capabilities are reported");

    assert_eq!(unsupported, ["vision_encoder", "speculative_decoding"]);
}

#[test]
fn capability_validation_uses_runtime_supported_set() {
    let metadata =
        load_metadata(&fixture("sample_metadata.yaml")).expect("valid YAML fixture parses");
    let runtime = RuntimeCapabilities {
        supported: vec!["kv_cache".to_string()],
    };

    let unsupported = validate(&metadata, &runtime).expect_err("missing GQA support is reported");

    assert_eq!(unsupported, ["grouped_query_attention"]);
}

#[test]
fn parses_and_validates_pipeline_fixture() {
    let spec = load_pipeline_spec(&crate_fixture("pipeline_valid.yaml"))
        .expect("valid pipeline fixture parses and validates");

    assert_eq!(spec.models.len(), 2);
    assert_eq!(
        spec.models["vision_encoder"].filename,
        "vision_encoder.onnx"
    );
    assert_eq!(spec.models["decoder"].role, "decoder");
    assert_eq!(
        spec.models["decoder"].tokenizer.as_deref(),
        Some("tokenizer.json")
    );
    assert_eq!(spec.dataflow[0].from, "vision_encoder.image_features");
    assert!(matches!(
        spec.strategy.kind,
        PipelineStrategyKind::Composite
    ));
    assert_eq!(spec.strategy.stages.len(), 2);
}

#[test]
fn pipeline_validation_rejects_dangling_edges() {
    let metadata = load_metadata(&crate_fixture("pipeline_dangling.yaml"))
        .expect("fixture parses structurally");
    let spec = metadata.pipeline.expect("pipeline section");
    let err = validate_pipeline_spec(&spec).expect_err("dangling component is rejected");

    assert!(
        err.errors
            .iter()
            .any(|error| error.contains("unknown component"))
    );
}

#[test]
fn pipeline_validation_rejects_cycles() {
    let metadata =
        load_metadata(&crate_fixture("pipeline_cycle.yaml")).expect("fixture parses structurally");
    let spec = metadata.pipeline.expect("pipeline section");
    let err = validate_pipeline_spec(&spec).expect_err("cycle is rejected");

    assert!(
        err.errors
            .iter()
            .any(|error| error.contains("contains a cycle"))
    );
}

#[test]
fn pipeline_vision_config_round_trips_via_json() {
    use onnx_genai_metadata::PipelineVisionConfig;

    let json = r#"{"image_placeholder_token_id": 32000, "tokens_per_tile": 256}"#;
    let decoded: PipelineVisionConfig = serde_json::from_str(json).expect("deserializes");
    assert_eq!(decoded.image_placeholder_token_id, Some(32000_i64));
    assert_eq!(decoded.tokens_per_tile, Some(256_usize));

    let round_tripped: PipelineVisionConfig =
        serde_json::from_str(&serde_json::to_string(&serde_json::from_str::<serde_json::Value>(json).unwrap()).unwrap())
            .expect("value round-trip");
    assert_eq!(round_tripped, decoded);
}

#[test]
fn pipeline_vision_config_round_trips_via_yaml() {
    use onnx_genai_metadata::PipelineVisionConfig;

    let yaml = "image_placeholder_token_id: -1\ntokens_per_tile: 64\n";
    let decoded: PipelineVisionConfig = serde_yaml::from_str(yaml).expect("deserializes");
    assert_eq!(decoded.image_placeholder_token_id, Some(-1_i64));
    assert_eq!(decoded.tokens_per_tile, Some(64_usize));
}

#[test]
fn pipeline_vision_config_optional_fields_round_trip_to_none() {
    use onnx_genai_metadata::PipelineVisionConfig;

    let json = r#"{}"#;
    let decoded: PipelineVisionConfig = serde_json::from_str(json).expect("deserializes");
    assert_eq!(decoded.image_placeholder_token_id, None);
    assert_eq!(decoded.tokens_per_tile, None);
}

#[test]
fn pipeline_spec_vision_field_round_trips() {
    use onnx_genai_metadata::PipelineVisionConfig;

    let yaml = "
pipeline:
  models:
    decoder:
      filename: decoder.onnx
      type: decoder
      tokenizer: tokenizer.json
  strategy:
    kind: autoregressive
    decoder: decoder
  vision:
    image_placeholder_token_id: 32000
    tokens_per_tile: 256
";
    let metadata: onnx_genai_metadata::InferenceMetadata =
        serde_yaml::from_str(yaml).expect("parses");
    let spec = metadata.pipeline.expect("pipeline section");
    let vision = spec.vision.expect("vision section");
    assert_eq!(
        vision,
        PipelineVisionConfig {
            image_placeholder_token_id: Some(32000),
            tokens_per_tile: Some(256),
        }
    );
}

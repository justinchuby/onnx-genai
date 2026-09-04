use std::path::Path;

use onnx_genai_metadata::{
    COMPRESSED_STATE_SCHEMA_VERSION, CompressedRecordFormat, CompressionRatio,
    StateGroupProperties, StateKind, StatePortRole, StateSemanticRole, StateUpdate,
    inference_metadata_schema_json, load_metadata_from_dir, parse_metadata, parse_metadata_json,
    resolve_state_plan, validate_metadata, version,
};

fn fixture() -> onnx_genai_metadata::InferenceMetadata {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../onnx-genai-engine/tests/fixtures/tiny-deepseek-v4-csa-schedule");
    load_metadata_from_dir(&root)
        .unwrap()
        .expect("compressed schedule metadata")
}

fn fixture_document() -> String {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../onnx-genai-engine/tests/fixtures/tiny-deepseek-v4-csa-schedule");
    std::fs::read_to_string(root.join("inference_metadata.yaml"))
        .expect("read compressed schedule metadata")
}

fn fixture_value() -> serde_json::Value {
    serde_yaml::from_str(&fixture_document()).expect("compressed schedule metadata is a JSON tree")
}

fn compressed_document() -> serde_json::Value {
    serde_json::json!({
        "schema_version": "v1.8",
        "pipeline": {
            "workflow": {
                "manifest": {},
                "publication_mode": "commit_only",
                "components": {},
                "steps": [],
                "serving": {
                    "active": "active",
                    "done": "done",
                    "state_service": {
                        "groups": {
                            "compressed_records.0": {
                                "kind": "compressed_attention",
                                "properties": {
                                    "kind": "compressed_attention",
                                    "ratio": "ratio4",
                                    "record_format": "fp8_e4m3_block64",
                                    "recurrence": "standard"
                                },
                                "sequence_axis": 1,
                                "layout": "batch_record_feature",
                                "aliasing": "forbidden",
                                "update": {
                                    "kind": "append"
                                },
                                "reuse": {
                                    "prefix_reusable": false,
                                    "evictable_prefix": false
                                },
                                "capabilities": {
                                    "snapshot": false,
                                    "fork": false
                                },
                                "ports": {
                                    "decoder": {
                                        "compressed_records.0.kv": {
                                            "input": "past_compressed_kv.0",
                                            "output": "present_compressed_kv.0",
                                            "role": "compressed_kv",
                                            "layer": 0
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    })
}

fn object_at_mut<'a>(
    document: &'a mut serde_json::Value,
    pointer: &str,
) -> &'a mut serde_json::Map<String, serde_json::Value> {
    document
        .pointer_mut(pointer)
        .unwrap_or_else(|| panic!("fixture has {pointer}"))
        .as_object_mut()
        .unwrap_or_else(|| panic!("{pointer} is an object"))
}

fn assert_rejection(
    label: &str,
    encoding: &str,
    error: onnx_genai_metadata::MetadataError,
    expected_path: &str,
    expected_reason: &str,
) {
    let reported = error.to_string();
    assert!(
        expected_path.is_empty() || reported.contains(expected_path),
        "{label} ({encoding}) did not report path {expected_path:?}: {reported}"
    );
    assert!(
        reported.contains(expected_reason),
        "{label} ({encoding}) did not report reason {expected_reason:?}: {reported}"
    );
}

fn assert_rejected_in_every_input_form(
    label: &str,
    document: &serde_json::Value,
    expected_path: &str,
    expected_reason: &str,
) {
    let yaml = serde_yaml::to_string(document).expect("test document serializes as YAML");
    for (encoding, content) in [
        ("YAML LF", yaml.clone()),
        ("YAML CRLF", yaml.replace('\n', "\r\n")),
    ] {
        let error = parse_metadata(&content, Some("yaml"))
            .err()
            .unwrap_or_else(|| panic!("{label} ({encoding}) must be rejected"));
        assert_rejection(label, encoding, error, expected_path, expected_reason);
    }

    let json = serde_json::to_string(document).expect("test document serializes as JSON");
    let error = parse_metadata(&json, Some("json"))
        .err()
        .unwrap_or_else(|| panic!("{label} (JSON text) must be rejected"));
    assert_rejection(label, "JSON text", error, expected_path, expected_reason);

    let error = parse_metadata_json(document)
        .err()
        .unwrap_or_else(|| panic!("{label} (JSON value) must be rejected"));
    assert_rejection(label, "JSON value", error, expected_path, expected_reason);
}

fn assert_accepted_in_every_input_form(label: &str, document: &serde_json::Value) {
    let yaml = serde_yaml::to_string(document).expect("test document serializes as YAML");
    for (encoding, content) in [
        ("YAML LF", yaml.clone()),
        ("YAML CRLF", yaml.replace('\n', "\r\n")),
    ] {
        parse_metadata(&content, Some("yaml"))
            .unwrap_or_else(|error| panic!("{label} ({encoding}) must parse: {error}"));
    }

    let json = serde_json::to_string(document).expect("test document serializes as JSON");
    parse_metadata(&json, Some("json"))
        .unwrap_or_else(|error| panic!("{label} (JSON text) must parse: {error}"));
    parse_metadata_json(document)
        .unwrap_or_else(|error| panic!("{label} (JSON value) must parse: {error}"));
}

fn state_service(
    metadata: &mut onnx_genai_metadata::InferenceMetadata,
) -> &mut onnx_genai_metadata::StateServiceContract {
    &mut metadata
        .pipeline
        .as_mut()
        .unwrap()
        .workflow
        .serving
        .as_mut()
        .unwrap()
        .state_service
}

fn matching_errors(metadata: &onnx_genai_metadata::InferenceMetadata, needle: &str) -> Vec<String> {
    validate_metadata(metadata)
        .unwrap_err()
        .into_iter()
        .filter(|error| error.contains(needle))
        .collect()
}

#[test]
fn canonical_schedule_lowers_exact_21_20_properties() {
    let metadata = fixture();
    assert_eq!(
        version::normalize(metadata.schema_version.as_deref()).unwrap(),
        COMPRESSED_STATE_SCHEMA_VERSION
    );
    validate_metadata(&metadata).expect("valid compressed state contract");
    let abi = metadata.decoder_io().expect("derived decoder ABI");
    assert!(
        abi.state_groups
            .iter()
            .filter(|group| group.kind == StateKind::CompressedAttention)
            .flat_map(|group| &group.ports)
            .all(|port| port.batch_axis == Some(0)),
        "the lowered ABI must carry the canonical request axis instead of guessing axis zero"
    );
    let properties = abi
        .state_groups
        .iter()
        .filter(|group| {
            group.kind == StateKind::CompressedAttention
                && matches!(group.update, Some(StateUpdate::Append))
        })
        .map(|group| group.properties.clone().expect("typed properties"))
        .collect::<Vec<_>>();
    assert_eq!(
        properties
            .iter()
            .filter(|properties| matches!(
                properties,
                StateGroupProperties::CompressedAttention {
                    ratio: CompressionRatio::Ratio4,
                    record_format: CompressedRecordFormat::Fp8E4m3Block64,
                    ..
                }
            ))
            .count(),
        21
    );
    assert_eq!(
        properties
            .iter()
            .filter(|properties| matches!(
                properties,
                StateGroupProperties::CompressedAttention {
                    ratio: CompressionRatio::Ratio128,
                    record_format: CompressedRecordFormat::F32,
                    ..
                }
            ))
            .count(),
        20
    );
}

#[test]
fn compressed_state_has_an_exact_v1_8_schema_floor() {
    let current = fixture_document();
    parse_metadata(&current, Some("yaml")).expect("v1.8 compressed state parses");

    let old = current.replacen("schema_version: v1.8", "schema_version: v1.7", 1);
    let error = parse_metadata(&old, Some("yaml"))
        .expect_err("v1.7 must refuse compressed-state vocabulary before typed parsing");
    let reported = error.to_string();
    assert!(
        reported.contains("compressed-attention")
            && reported.contains("authored schema version v1.7")
            && reported.contains("minimum schema version v1.8")
            && !reported.contains("unknown variant"),
        "{reported}"
    );

    let future = current.replacen("schema_version: v1.8", "schema_version: v1.9", 1);
    let error =
        parse_metadata(&future, Some("yaml")).expect_err("unknown future versions fail closed");
    let reported = error.to_string();
    assert!(
        reported.contains("schema version v1.9")
            && reported.contains("reads up to v1.8")
            && !reported.contains("unknown variant"),
        "{reported}"
    );
}

#[test]
fn typed_admission_applies_the_same_compressed_state_version_gate() {
    let old = fixture_document().replacen("schema_version: v1.8", "schema_version: v1.7", 1);
    let metadata: onnx_genai_metadata::InferenceMetadata =
        serde_yaml::from_str(&old).expect("typed serde retains compressed-state vocabulary");
    let errors = validate_metadata(&metadata).expect_err("typed admission must apply v1.8 floor");
    let matching = errors
        .iter()
        .filter(|error| error.contains("minimum schema version v1.8"))
        .collect::<Vec<_>>();
    assert_eq!(matching.len(), 1, "{errors:#?}");
}

#[test]
fn malformed_and_unknown_compressed_fields_fail_closed() {
    for (label, pointer, value, path, reason) in [
        (
            "unknown group kind",
            "/pipeline/workflow/serving/state_service/groups/compressed_records.0/kind",
            serde_json::json!("compressed_attention_future"),
            "state_service.groups",
            "unknown variant",
        ),
        (
            "unknown state role",
            "/pipeline/workflow/serving/state_service/groups/compressed_records.0/ports/decoder/compressed_records.0.kv/role",
            serde_json::json!("compressed_kv_future"),
            "ports.decoder",
            "unknown variant",
        ),
        (
            "unknown compression ratio",
            "/pipeline/workflow/serving/state_service/groups/compressed_records.0/properties/ratio",
            serde_json::json!("ratio16"),
            "properties",
            "unknown variant",
        ),
        (
            "malformed port layer",
            "/pipeline/workflow/serving/state_service/groups/compressed_records.0/ports/decoder/compressed_records.0.kv/layer",
            serde_json::json!("second"),
            "ports.decoder",
            "invalid type",
        ),
        (
            "malformed sequence axis",
            "/pipeline/workflow/serving/state_service/groups/compressed_records.0/sequence_axis",
            serde_json::json!("records"),
            "state_service.groups",
            "invalid type",
        ),
        (
            "malformed compression recurrence",
            "/pipeline/workflow/serving/state_service/groups/compressed_records.0/properties/recurrence",
            serde_json::json!(4),
            "properties",
            "invalid type",
        ),
        (
            "malformed snapshot capability",
            "/pipeline/workflow/serving/state_service/groups/compressed_records.0/capabilities/snapshot",
            serde_json::json!("eventually"),
            "capabilities",
            "invalid type",
        ),
    ] {
        let mut document = compressed_document();
        *document
            .pointer_mut(pointer)
            .unwrap_or_else(|| panic!("fixture has {pointer}")) = value;
        assert_rejected_in_every_input_form(label, &document, path, reason);
    }

    let mut missing = compressed_document();
    object_at_mut(
        &mut missing,
        "/pipeline/workflow/serving/state_service/groups/compressed_records.0/properties",
    )
    .remove("record_format")
    .expect("fixture declares record_format");
    assert_rejected_in_every_input_form(
        "missing record format",
        &missing,
        "properties",
        "missing field `record_format`",
    );
}

#[test]
fn unknown_fields_fail_closed_at_every_compressed_state_object_boundary() {
    let cases = [
        ("metadata root", "", "future_metadata", ""),
        (
            "state service",
            "/pipeline/workflow/serving/state_service",
            "future_state_service",
            "state_service",
        ),
        (
            "state group",
            "/pipeline/workflow/serving/state_service/groups/compressed_records.0",
            "future_group_layout",
            "state_service.groups",
        ),
        (
            "compressed properties",
            "/pipeline/workflow/serving/state_service/groups/compressed_records.0/properties",
            "future_compression_codec",
            "properties",
        ),
        (
            "state reuse",
            "/pipeline/workflow/serving/state_service/groups/compressed_records.0/reuse",
            "future_reuse_scope",
            "reuse",
        ),
        (
            "state capabilities",
            "/pipeline/workflow/serving/state_service/groups/compressed_records.0/capabilities",
            "future_snapshot_mode",
            "capabilities",
        ),
        (
            "state port member",
            "/pipeline/workflow/serving/state_service/groups/compressed_records.0/ports/decoder/compressed_records.0.kv",
            "future_port_binding",
            "ports.decoder",
        ),
    ];

    for (label, pointer, field, path) in cases {
        let mut document = compressed_document();
        assert!(
            object_at_mut(&mut document, pointer)
                .insert(field.to_string(), serde_json::json!("unsupported"))
                .is_none(),
            "fixture unexpectedly declares {field}"
        );
        assert_rejected_in_every_input_form(label, &document, path, field);
    }

    for kind in ["append", "replace"] {
        let mut update = compressed_document();
        let object = object_at_mut(
            &mut update,
            "/pipeline/workflow/serving/state_service/groups/compressed_records.0/update",
        );
        object.insert("kind".to_string(), serde_json::json!(kind));
        object.insert(
            "future_update_mode".to_string(),
            serde_json::json!("unsupported"),
        );
        assert_rejected_in_every_input_form(
            &format!("{kind} state update"),
            &update,
            "update",
            "future_update_mode",
        );
    }

    let mut source = compressed_document();
    object_at_mut(
        &mut source,
        "/pipeline/workflow/serving/state_service/groups/compressed_records.0",
    )
    .insert(
        "source".to_string(),
        serde_json::json!({"path": "../../foreign-state.bin"}),
    );
    assert_rejected_in_every_input_form(
        "arbitrary state source",
        &source,
        "state_service.groups",
        "source",
    );

    let mut checkpoint = compressed_document();
    let group = object_at_mut(
        &mut checkpoint,
        "/pipeline/workflow/serving/state_service/groups/compressed_records.0",
    );
    group.insert(
        "checkpoint".to_string(),
        serde_json::json!({
            "adapter": "example.invalid/checkpoint",
            "version": "37",
            "future_checkpoint_transport": "shared_memory"
        }),
    );
    assert_rejected_in_every_input_form(
        "state checkpoint extension",
        &checkpoint,
        "checkpoint",
        "future_checkpoint_transport",
    );
}

#[test]
fn known_fields_and_normative_extension_surfaces_remain_accepted() {
    let document = compressed_document();
    assert_accepted_in_every_input_form("known compressed-state fields", &document);

    let mut ignorable_profile = fixture_value();
    object_at_mut(&mut ignorable_profile, "").insert(
        "profiles".to_string(),
        serde_json::json!({
            "future_task": {
                "kind": "com.example.future-task",
                "version": "1",
                "requirement": "ignorable"
            }
        }),
    );
    assert_accepted_in_every_input_form("normative ignorable profile", &ignorable_profile);
    let metadata = parse_metadata_json(&ignorable_profile).expect("ignorable profile parses");
    validate_metadata(&metadata).expect("normative ignorable profile remains skippable");

    let mut checkpoint = compressed_document();
    object_at_mut(
        &mut checkpoint,
        "/pipeline/workflow/serving/state_service/groups/compressed_records.0",
    )
    .insert(
        "checkpoint".to_string(),
        serde_json::json!({
            "adapter": "example.invalid/checkpoint",
            "version": "37"
        }),
    );
    assert_accepted_in_every_input_form("typed checkpoint extension", &checkpoint);
    let metadata = parse_metadata_json(&checkpoint).expect("checkpoint declaration parses");
    let errors = validate_metadata(&metadata)
        .expect_err("an unregistered checkpoint pair must fail at extension admission")
        .join("; ");
    assert!(
        errors.contains("compressed_records.0")
            && errors.contains("example.invalid/checkpoint@37")
            && errors.contains("onnx-genai.kv-checkpoint@1"),
        "{errors}"
    );
}

#[test]
fn published_schema_closes_every_compressed_state_object() {
    let schema: serde_json::Value =
        serde_json::from_str(&inference_metadata_schema_json().expect("schema serializes"))
            .expect("schema is JSON");
    assert_eq!(schema["additionalProperties"], serde_json::json!(false));

    for definition in [
        "ServingServiceContract",
        "StateServiceContract",
        "StateGroupContract",
        "StateCheckpointContract",
        "StateReuse",
        "StateGroupCapabilities",
        "StatePortAlias",
    ] {
        assert_eq!(
            schema["$defs"][definition]["additionalProperties"],
            serde_json::json!(false),
            "{definition} must reject fields outside the typed core schema"
        );
    }

    for definition in ["StateGroupProperties", "StateUpdate"] {
        let variants = schema["$defs"][definition]["oneOf"]
            .as_array()
            .unwrap_or_else(|| panic!("{definition} has tagged object variants"));
        assert!(
            variants
                .iter()
                .all(|variant| variant["additionalProperties"] == serde_json::json!(false)),
            "{definition} variants must reject fields outside the typed core schema: {variants:#?}"
        );
    }
}

#[test]
fn compressed_records_and_carries_are_attention_state() {
    let metadata = fixture();
    let workflow = &metadata
        .pipeline
        .as_ref()
        .expect("fixture declares pipeline")
        .workflow;
    let plan = resolve_state_plan(workflow);

    for state in ["compressed_records.0.kv", "compressed_carries.0.kv"] {
        assert_eq!(
            plan.cell(state)
                .expect("compressed state is planned")
                .semantic_role,
            StateSemanticRole::AttentionKv,
            "{state}"
        );
    }
}

#[test]
fn compressed_group_without_properties_is_rejected() {
    let mut metadata = fixture();
    let group = state_service(&mut metadata)
        .groups
        .get_mut("compressed_records.0")
        .unwrap();
    group.properties = None;
    let errors = validate_metadata(&metadata).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.contains("no compressed_attention properties"))
    );
}

#[test]
fn scalar_compressed_carry_is_rejected_by_schema() {
    let mut metadata = fixture();
    metadata
        .pipeline
        .as_mut()
        .unwrap()
        .workflow
        .state
        .get_mut("compressed_carries.0.kv")
        .unwrap()
        .contract
        .shape
        .clear();
    let errors = validate_metadata(&metadata).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.contains("scalar rank 0") && error.contains("row-scoped state")),
        "{errors:#?}"
    );
}

#[test]
fn compressed_record_group_cannot_use_replace() {
    let mut metadata = fixture();
    let group = state_service(&mut metadata)
        .groups
        .get_mut("compressed_records.0")
        .unwrap();
    group.update = Some(StateUpdate::Replace);
    group.sequence_axis = None;
    let errors = validate_metadata(&metadata).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| { error.contains("record roles append and carry roles replace") })
    );
}

#[test]
fn fp4_is_not_a_compressed_kv_record_format() {
    let mut metadata = fixture();
    let group = state_service(&mut metadata)
        .groups
        .get_mut("compressed_records.0")
        .unwrap();
    group.properties = Some(StateGroupProperties::CompressedAttention {
        ratio: CompressionRatio::Ratio4,
        record_format: CompressedRecordFormat::Fp4E2m1Block32,
        recurrence: onnx_genai_metadata::CompressionRecurrence::Standard,
    });
    let errors = validate_metadata(&metadata).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.contains("FP4 is reserved for index_key records"))
    );
}

#[test]
fn unsupported_compression_recurrence_is_rejected() {
    let mut metadata = fixture();
    let group = state_service(&mut metadata)
        .groups
        .get_mut("compressed_records.0")
        .unwrap();
    group.properties = Some(StateGroupProperties::CompressedAttention {
        ratio: CompressionRatio::Ratio4,
        record_format: CompressedRecordFormat::Fp8E4m3Block64,
        recurrence: onnx_genai_metadata::CompressionRecurrence::MultiTokenPrediction,
    });
    let errors = validate_metadata(&metadata).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.contains("multi_token_prediction")),
        "{errors:#?}"
    );
}

#[test]
fn compressed_ports_require_role_and_layer() {
    for missing in ["role", "layer"] {
        let mut metadata = fixture();
        let alias = state_service(&mut metadata)
            .groups
            .get_mut("compressed_records.0")
            .unwrap()
            .ports
            .get_mut("decoder")
            .unwrap()
            .get_mut("compressed_records.0.kv")
            .unwrap();
        match missing {
            "role" => alias.role = None,
            "layer" => alias.layer = None,
            _ => unreachable!(),
        }
        let errors = validate_metadata(&metadata).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("compressed-attention") && error.contains(missing)),
            "{missing}: {errors:#?}"
        );
    }
}

#[test]
fn duplicate_layer_role_is_reported_once_by_the_property_index() {
    let mut metadata = fixture();
    let alias = state_service(&mut metadata)
        .groups
        .get_mut("compressed_records.0")
        .unwrap()
        .ports
        .get_mut("decoder")
        .unwrap()
        .get_mut("compressed_records.0.index")
        .unwrap();
    alias.role = Some(StatePortRole::CompressedKv);

    let errors = matching_errors(&metadata, "declares role 'CompressedKv' more than once");
    assert_eq!(errors.len(), 1, "{errors:#?}");
}

#[test]
fn contradictory_layer_properties_are_reported_once_by_the_property_index() {
    let mut metadata = fixture();
    state_service(&mut metadata)
        .groups
        .get_mut("compressed_carries.0")
        .unwrap()
        .properties = Some(StateGroupProperties::CompressedAttention {
        ratio: CompressionRatio::Ratio128,
        record_format: CompressedRecordFormat::F32,
        recurrence: onnx_genai_metadata::CompressionRecurrence::Standard,
    });

    let errors = matching_errors(&metadata, "contradictory properties across state groups");
    assert_eq!(errors.len(), 1, "{errors:#?}");
}

#[test]
fn incomplete_layer_roles_are_reported_once_by_the_property_index() {
    let mut metadata = fixture();
    let service = state_service(&mut metadata);
    service
        .groups
        .get_mut("compressed_records.0")
        .unwrap()
        .ports
        .get_mut("decoder")
        .unwrap()
        .remove("compressed_records.0.index");
    service
        .groups
        .get_mut("compressed_carries.0")
        .unwrap()
        .ports
        .get_mut("decoder")
        .unwrap()
        .remove("compressed_carries.0.index");

    let errors = matching_errors(&metadata, "with ratio 'Ratio4' has incomplete roles");
    assert_eq!(errors.len(), 1, "{errors:#?}");
}

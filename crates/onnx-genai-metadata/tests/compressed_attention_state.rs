use std::path::Path;

use onnx_genai_metadata::{
    COMPRESSED_STATE_SCHEMA_VERSION, CompressedRecordFormat, CompressionRatio,
    StateGroupProperties, StateKind, StatePortRole, StateSemanticRole, StateUpdate,
    load_metadata_from_dir, parse_metadata, resolve_state_plan, validate_metadata, version,
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
    let document = fixture_document();
    for (label, malformed, needle) in [
        (
            "unknown group kind",
            document.replacen(
                "kind: compressed_attention",
                "kind: compressed_attention_future",
                1,
            ),
            "unknown variant",
        ),
        (
            "unknown state role",
            document.replacen("role: compressed_kv", "role: compressed_kv_future", 1),
            "unknown variant",
        ),
        (
            "unknown compression ratio",
            document.replacen("ratio: ratio4", "ratio: ratio16", 1),
            "unknown variant",
        ),
        (
            "missing record format",
            document.replacen("record_format: fp8_e4m3_block64", "record_format_missing: true", 1),
            "unknown field",
        ),
        (
            "unknown group field",
            document.replacen(
                "            kind: compressed_attention\n            properties:",
                "            kind: compressed_attention\n            future_layout: tiled\n            properties:",
                1,
            ),
            "unknown field",
        ),
        (
            "arbitrary state source",
            document.replacen(
                "            kind: compressed_attention\n            properties:",
                "            kind: compressed_attention\n            source:\n              path: ../../foreign-state.bin\n            properties:",
                1,
            ),
            "unknown field",
        ),
    ] {
        let error = match parse_metadata(&malformed, Some("yaml")) {
            Ok(_) => panic!("{label} must be rejected"),
            Err(error) => error,
        };
        assert!(error.to_string().contains(needle), "{label}: {error}");
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

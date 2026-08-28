use std::path::Path;

use onnx_genai_metadata::{
    CompressedRecordFormat, CompressionRatio, StateGroupProperties, StateKind, StatePortRole,
    StateSemanticRole, StateUpdate, load_metadata_from_dir, resolve_state_plan, validate_metadata,
};

fn fixture() -> onnx_genai_metadata::InferenceMetadata {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../onnx-genai-engine/tests/fixtures/tiny-deepseek-v4-csa-schedule");
    load_metadata_from_dir(&root)
        .unwrap()
        .expect("compressed schedule metadata")
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
    validate_metadata(&metadata).expect("valid compressed state contract");
    let abi = metadata.decoder_io().expect("derived decoder ABI");
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

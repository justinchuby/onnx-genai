#![deny(warnings)]

use onnx_genai_metadata::StateUpdate;

fn classify_legacy_update(update: StateUpdate) -> &'static str {
    match update {
        StateUpdate::Append => "append",
        StateUpdate::Replace => "replace",
        StateUpdate::IndexedScatter {
            write_indices: _,
            capacity: _,
            write_indices_ports: _,
            kv_length_ports: _,
        } => "indexed_scatter",
    }
}

#[test]
fn downstream_unit_construction_and_exhaustive_matching_remain_source_compatible() {
    assert_eq!(classify_legacy_update(StateUpdate::Append), "append");
    assert_eq!(classify_legacy_update(StateUpdate::Replace), "replace");
}

#[test]
fn unit_variant_serialization_remains_wire_compatible() {
    assert_eq!(
        serde_json::to_value(StateUpdate::Append).unwrap(),
        serde_json::json!({"kind": "append"})
    );
    assert_eq!(
        serde_json::to_value(StateUpdate::Replace).unwrap(),
        serde_json::json!({"kind": "replace"})
    );
}

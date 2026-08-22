//! Fixed-size recurrent state uses one generic replacement discipline.

use onnx_genai_metadata::{
    InferenceMetadata, StateGroupContract, StateKind, StateUpdate, validate_metadata,
};

#[test]
fn recurrent_replace_group_needs_no_sequence_axis() {
    let yaml = r#"
kind: recurrent
layout: bhd
update:
  kind: replace
"#;
    let group: StateGroupContract = serde_yaml::from_str(yaml).expect("state group parses");
    assert_eq!(group.kind, StateKind::Recurrent);
    assert_eq!(group.sequence_axis, None);
    assert_eq!(group.update, Some(StateUpdate::Replace));

    let round_trip = serde_yaml::to_string(&group).expect("state group serializes");
    assert!(round_trip.contains("kind: recurrent"));
    assert!(round_trip.contains("kind: replace"));
    assert!(!round_trip.contains("sequence_axis"));
}

#[test]
fn sequence_state_keeps_an_explicit_axis() {
    let yaml = r#"
kind: sliding_attention
sequence_axis: 2
layout: bnsh
update:
  kind: append
"#;
    let group: StateGroupContract = serde_yaml::from_str(yaml).expect("state group parses");
    assert_eq!(group.sequence_axis, Some(2));
    assert_eq!(group.update, Some(StateUpdate::Append));
}

#[test]
fn replace_state_rejects_a_sequence_axis() {
    let yaml = include_str!(
        "../../../examples/inference_metadata/catalogue/16-linear-attention-recurrent.yaml"
    )
    .replace("\r\n", "\n")
    .replacen(
        "kind: recurrent\n            layout: bhfv",
        "kind: recurrent\n            sequence_axis: 2\n            layout: bhfv",
        1,
    );
    let metadata: InferenceMetadata = serde_yaml::from_str(&yaml).expect("metadata parses");
    let errors = validate_metadata(&metadata).expect_err("replace state with an axis is invalid");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("replace") && error.contains("sequence_axis")),
        "unexpected validation errors: {errors:?}"
    );
}

//! Versioned external preprocessing output binding, independent of serving.
use onnx_genai_metadata::{parse_metadata, validate_metadata};
use serde_json::{Value, json};
fn package() -> Value {
    serde_json::from_str(include_str!("../../../tests/fixtures/onnx_genai_workflows/transcription_features/inference_metadata.yaml")).unwrap()
}
fn admit(value: &Value) -> Result<(), String> {
    let metadata =
        parse_metadata(&value.to_string(), Some("json")).map_err(|error| error.to_string())?;
    validate_metadata(&metadata).map_err(|errors| format!("{errors:?}"))
}
#[test]
fn external_audio_feature_binding_requires_its_schema_version_and_named_media_role() {
    let value = package();
    admit(&value).unwrap();
    let mut old = value.clone();
    old["schema_version"] = json!("v1.15");
    let error = admit(&old).unwrap_err().to_string();
    assert!(
        error.contains("v1.16") && error.contains("feature input"),
        "{error}"
    );
    for (field, value) in [
        ("dtype", json!("int64")),
        ("content", json!("waveform")),
        ("optional", json!(true)),
    ] {
        let mut bad = package();
        bad["preprocessing"]["audio"]["outputs"][0][field] = value;
        assert!(admit(&bad).is_err(), "{field}");
    }
    let mut bad = package();
    bad["preprocessing"]["audio"]["outputs"][0]["name"] = json!("not-bound");
    assert!(admit(&bad).is_err());
    let mut bad = package();
    bad["pipeline"]["workflow"]["inputs"]["request.features"]["role"]["role"] =
        json!("prompt_text");
    assert!(admit(&bad).is_err());
}
#[test]
fn external_audio_feature_output_contract_must_match_its_input() {
    for shape in [json!([1, 7, 4]), json!([1, 7, 5])] {
        let mut value = package();
        value["preprocessing"]["audio"]["outputs"][0]["contract"] =
            json!({"dtype":"float32","shape":shape});
        let result = admit(&value);
        assert_eq!(result.is_ok(), shape == json!([1, 7, 4]), "{result:?}");
    }
}

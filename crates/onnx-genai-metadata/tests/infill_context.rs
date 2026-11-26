use onnx_genai_metadata::{parse_metadata_json, validate_metadata};
use serde_json::json;
fn document() -> serde_json::Value {
    json!({"schema_version":"1.18","package":{"infill":{"prefix_token":"<P>","middle_token":"<M>","suffix_token":"<S>","order":"spm","extra_context_template":"{% for file in input_extra %}{{file.filename}}:{{file.text}}{% endfor %}"}}})
}
#[test]
fn infill_contract_requires_its_version_and_preserves_explicit_order_and_template() {
    let metadata = parse_metadata_json(&document()).unwrap();
    validate_metadata(&metadata).unwrap();
    let spec = metadata.package.as_ref().unwrap().infill.as_ref().unwrap();
    assert_eq!(spec.order, onnx_genai_metadata::InfillOrder::Spm);
    assert!(
        spec.extra_context_template
            .as_ref()
            .unwrap()
            .contains("filename")
    );
    let mut old = document();
    old["schema_version"] = json!("1.17");
    let error = parse_metadata_json(&old).unwrap_err();
    assert!(error.to_string().contains("package.infill"));
    assert!(error.to_string().contains("1.18"));
}
#[test]
fn infill_contract_rejects_ambiguous_roles_and_empty_templates() {
    for (field, value) in [
        ("prefix_token", json!("<M>")),
        ("middle_token", json!("")),
        ("extra_context_template", json!("  ")),
        ("additional_stop_tokens", json!([""])),
    ] {
        let mut invalid = document();
        invalid["package"]["infill"][field] = value;
        let metadata = parse_metadata_json(&invalid).unwrap();
        assert!(validate_metadata(&metadata).is_err(), "{field}");
    }
}

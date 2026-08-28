use onnx_genai_metadata::{parse_metadata, validate_metadata};

#[test]
fn tool_protocol_declares_one_exact_identity_and_version() {
    let metadata = parse_metadata(
        r#"
schema_version: v1.3
package:
  tool_protocol:
    identity: tagged-json
    version: v1
"#,
        Some("yaml"),
    )
    .expect("metadata parses");
    let protocol = metadata
        .package
        .as_ref()
        .and_then(|package| package.tool_protocol.as_ref())
        .expect("tool protocol is preserved");
    assert_eq!(protocol.identity, "tagged-json");
    assert_eq!(protocol.version, "v1");
    validate_metadata(&metadata).expect("exact declaration validates");
}

#[test]
fn tool_protocol_refuses_boolean_capability_and_ambiguous_values() {
    let error = parse_metadata(
        r#"
package:
  tool_protocol:
    supports_tools: true
"#,
        Some("yaml"),
    )
    .expect_err("a boolean does not declare a protocol")
    .to_string();
    assert!(error.contains("supports_tools"), "{error}");

    let metadata = parse_metadata(
        r#"
schema_version: v1.3
package:
  tool_protocol:
    identity: " "
    version: " "
"#,
        Some("yaml"),
    )
    .expect("schema accepts strings so validator can explain the defect");
    let errors = validate_metadata(&metadata).expect_err("blank declarations are ambiguous");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("package.tool_protocol.identity")),
        "{errors:#?}"
    );
    assert!(
        errors
            .iter()
            .any(|error| error.contains("package.tool_protocol.version")),
        "{errors:#?}"
    );
}

#[test]
fn tool_protocol_requires_its_introducing_schema_version() {
    let metadata = parse_metadata(
        r#"
schema_version: v1.2
package:
  tool_protocol:
    identity: tagged-json
    version: v1
"#,
        Some("yaml"),
    )
    .expect("older supported schema parses before structural validation");
    let errors = validate_metadata(&metadata).expect_err("the field needs schema v1.3");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("package.tool_protocol") && error.contains("v1.3")),
        "{errors:#?}"
    );
}

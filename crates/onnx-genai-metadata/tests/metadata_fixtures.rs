use onnx_genai_metadata::{InferenceMetadata, validate_metadata};

#[test]
fn minimal_workflow_document_is_valid() {
    let metadata: InferenceMetadata = serde_yaml::from_str(
        r#"
schema_version: v1
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets:
        ai.onnx: 24
      capabilities: [workflow_ssa, linear_effects]
    components:
      noop:
        implementation:
          kind: binding
        ports: {}
    steps:
      - kind: invoke
        component: noop
"#,
    )
    .expect("workflow metadata parses");
    validate_metadata(&metadata).expect("minimal workflow validates");
}

#[test]
fn legacy_pipeline_control_fields_are_rejected() {
    let error = serde_yaml::from_str::<InferenceMetadata>(
        r#"
pipeline:
  strategy:
    kind: autoregressive
  phases: {}
"#,
    )
    .expect_err("legacy strategy/phases must not deserialize");
    let message = error.to_string();
    assert!(message.contains("unknown field") || message.contains("missing field `workflow`"));
}

#[test]
fn serialized_compiler_bookkeeping_is_rejected() {
    for field in [
        "graph: { kind: sequence, steps: [] }",
        "initial_effects: { stream: stream.0 }",
    ] {
        let document = format!(
            r#"
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: {{ ai.onnx: 24 }}
      capabilities: []
    components: {{}}
    steps: []
    {field}
"#
        );
        let error = serde_yaml::from_str::<InferenceMetadata>(&document)
            .expect_err("compiler bookkeeping must not deserialize");
        assert!(error.to_string().contains("unknown field"));
    }
}

#[test]
fn emit_valid_length_requires_integer_scalar_or_vector() {
    fn errors(dtype: &str, rank: usize, shape: &str) -> Vec<String> {
        let metadata: InferenceMetadata = serde_yaml::from_str(&format!(
            r#"
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: {{ ai.onnx: 24 }}
      capabilities: [workflow_ssa, linear_effects, typed_emit, emit_valid_length]
    inputs:
      value:
        contract: {{ dtype: int64, rank: 1, shape: [sequence] }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: value }}
        required: true
      length:
        contract: {{ dtype: {dtype}, rank: {rank}, shape: {shape} }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: length }}
        required: true
    outputs:
      result:
        contract: {{ dtype: int64, rank: 1, shape: [valid] }}
        role: tokens
        stage: pre_adapter
    components: {{}}
    steps:
      - kind: emit
        value: value
        valid_length: length
        output: result
        mode: replace
"#
        ))
        .expect("workflow metadata parses");
        validate_metadata(&metadata).expect_err("invalid valid_length must fail")
    }

    assert!(
        errors("float32", 0, "[]")
            .iter()
            .any(|error| error.contains("must have an integer dtype"))
    );
    assert!(
        errors("int64", 2, "[batch, one]")
            .iter()
            .any(|error| error.contains("must be a scalar or rank-one tensor"))
    );
}

#[test]
fn advisory_state_cannot_be_session_persistent() {
    let metadata: InferenceMetadata = serde_yaml::from_str(
        r#"
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: { ai.onnx: 24 }
      capabilities: [workflow_ssa, linear_effects, advisory_state]
    inputs:
      estimate:
        contract: { dtype: float32, rank: 1, shape: [batch] }
        role: { kind: opaque }
        source: { kind: application, name: estimate }
        required: true
    outputs: {}
    components:
      noop:
        implementation: { kind: binding }
        ports: {}
    state:
      estimate:
        contract: { dtype: float32, rank: 1, shape: [batch] }
        class: advisory
        scope: session
        initializer: estimate
        recurrence: { kind: invariant }
    steps:
      - kind: invoke
        component: noop
"#,
    )
    .expect("workflow metadata parses");
    let errors = validate_metadata(&metadata).expect_err("advisory session state must fail");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("must use invocation scope")),
        "{errors:?}"
    );
}

#[test]
fn versioned_adapter_contract_rejects_unknown_action() {
    let metadata: InferenceMetadata = serde_yaml::from_str(
        r#"
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: { ai.onnx: 24 }
      adapter_abis: { onnx-genai.grammar-guidance: "1" }
      capabilities: []
    components:
      grammar:
        implementation:
          kind: adapter
          abi: onnx-genai.grammar-guidance
          version: "1"
        contract:
          id: onnx-genai.grammar-guidance
          version: "1"
          parameters: { action: typo }
    steps: []
"#,
    )
    .expect("workflow metadata parses");
    let errors = validate_metadata(&metadata).expect_err("unknown adapter action must fail");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("unsupported action")),
        "{errors:?}"
    );
}

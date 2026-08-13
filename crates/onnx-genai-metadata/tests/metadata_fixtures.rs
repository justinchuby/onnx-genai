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
      capabilities: [workflow_ssa]
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
fn removed_top_level_execution_surfaces_are_rejected() {
    for field in [
        "strategy",
        "structured_output",
        "generation",
        "tokens",
        "speculative",
        "speculator_config",
    ] {
        let document = format!("{field}: {{}}\n");
        let error = serde_yaml::from_str::<InferenceMetadata>(&document)
            .expect_err("removed top-level execution metadata must not deserialize");
        assert!(error.to_string().contains("unknown field"), "{error}");
    }
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
      capabilities: [workflow_ssa, typed_emit, emit_valid_length]
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
fn advisory_state_may_be_session_scoped_but_is_not_semantic() {
    let metadata: InferenceMetadata = serde_yaml::from_str(
        r#"
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: { ai.onnx: 24 }
      capabilities: [workflow_ssa, advisory_state, session_state_lease]
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
    validate_metadata(&metadata).expect("advisory session state is resettable and non-semantic");
}

#[test]
fn kv_state_binds_named_runtime_service_group() {
    let metadata: InferenceMetadata = serde_yaml::from_str(
        r#"
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: { ai.onnx: 24 }
      capabilities: [workflow_ssa, serving_service_contract]
    inputs:
      active:
        contract: { dtype: bool, rank: 1, shape: [batch] }
        role: { kind: opaque }
        source: { kind: application, name: active }
      done:
        contract: { dtype: bool, rank: 1, shape: [batch] }
        role: { kind: opaque }
        source: { kind: application, name: done }
      slot_ids:
        contract: { dtype: int64, rank: 1, shape: [batch] }
        role: { kind: opaque }
        source: { kind: application, name: slot_ids }
      accepted_len:
        contract: { dtype: int64, rank: 1, shape: [batch] }
        role: { kind: opaque }
        source: { kind: application, name: accepted_len }
      cache_lengths:
        contract: { dtype: int64, rank: 1, shape: [batch] }
        role: { kind: opaque }
        source: { kind: application, name: cache_lengths }
      empty_cache:
        contract: { dtype: float16, rank: 4, shape: [batch, heads, sequence, head_dim] }
        role: { kind: opaque }
        source: { kind: application, name: empty_cache }
    components:
      decoder:
        implementation: { kind: onnx, artifact: decoder.onnx }
        ports: {}
    state:
      cache:
        contract: { dtype: float16, rank: 4, shape: [batch, heads, sequence, head_dim] }
        class: semantic
        scope: invocation
        initializer: empty_cache
        recurrence: { kind: invariant }
        service_group: decoder_cache
      cache_lengths:
        contract: { dtype: int64, rank: 1, shape: [batch] }
        class: semantic
        scope: invocation
        initializer: cache_lengths
        recurrence: { kind: invariant }
    serving:
      active: active
      done: done
      accepted_len: accepted_len
      slot_ids: slot_ids
      kv_service:
        paging: paged
        allocation: runtime
        groups:
          decoder_cache:
            sequence_axis: 2
            layout: bnsh
            logical_lengths: cache_lengths
            storage: shared_buffer
            ports:
              decoder:
                cache: { input: past_key_values, output: present_key_values }
    steps:
      - kind: invoke
        component: decoder
"#,
    )
    .expect("KV service metadata parses");
    validate_metadata(&metadata).expect("KV service group is executable and bound");

    let mut missing_lengths = metadata.clone();
    missing_lengths
        .pipeline
        .as_mut()
        .expect("pipeline")
        .workflow
        .state
        .remove("cache_lengths");
    let errors = validate_metadata(&missing_lengths).expect_err("logical lengths are required");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("unknown logical_lengths state")),
        "{errors:?}"
    );

    let mut missing_accepted = metadata;
    missing_accepted
        .pipeline
        .as_mut()
        .expect("pipeline")
        .workflow
        .serving
        .as_mut()
        .expect("serving")
        .accepted_len = None;
    let errors = validate_metadata(&missing_accepted).expect_err("accepted lengths are required");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("accepted_len is required")),
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

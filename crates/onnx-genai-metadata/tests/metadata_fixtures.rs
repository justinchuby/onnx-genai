use onnx_genai_metadata::{InferenceMetadata, WorkflowNode, compile_workflow, validate_metadata};

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
fn optional_input_presence_is_an_explicit_branch_predicate() {
    let metadata: InferenceMetadata = serde_yaml::from_str(
        r#"
schema_version: v1
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: { ai.onnx: 24 }
      capabilities: [workflow_ssa, nested_control_flow, input_presence]
    inputs:
      request.image:
        contract: { dtype: uint8, rank: 1, shape: [encoded_bytes] }
        role: { kind: runtime, version: "1.0", role: media }
        source: { kind: request }
        required: false
        present_as: request.image_present
    components:
      noop:
        implementation: { kind: binding }
        ports: {}
    steps:
      - kind: branch
        predicate: request.image_present
        cases:
          "true": { kind: invoke, component: noop }
        default: { kind: invoke, component: noop }
"#,
    )
    .expect("workflow metadata parses");
    validate_metadata(&metadata).expect("explicit input presence validates");
}

#[test]
fn optional_tensor_without_default_or_presence_is_rejected() {
    let metadata: InferenceMetadata = serde_yaml::from_str(
        r#"
schema_version: v1
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: { ai.onnx: 24 }
      capabilities: [workflow_ssa]
    inputs:
      request.image:
        contract: { dtype: uint8, rank: 1, shape: [encoded_bytes] }
        role: { kind: runtime, version: "1.0", role: media }
        source: { kind: request }
        required: false
    components: {}
    steps: []
"#,
    )
    .expect("workflow metadata parses");
    let errors = validate_metadata(&metadata).expect_err("implicit absence must fail");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("literal default or present_as predicate"))
    );
}

#[test]
fn optional_tensor_must_only_be_read_in_its_presence_branch() {
    let metadata: InferenceMetadata = serde_yaml::from_str(
        r#"
schema_version: v1
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: { ai.onnx: 24 }
      capabilities: [workflow_ssa, input_presence]
    inputs:
      request.image:
        contract: { dtype: uint8, rank: 1, shape: [encoded_bytes] }
        role: { kind: runtime, version: "1.0", role: media }
        source: { kind: request }
        required: false
        present_as: request.image_present
    components:
      consume:
        implementation: { kind: binding }
        ports:
          inputs:
            image: { dtype: uint8, rank: 1, shape: [encoded_bytes] }
    steps:
      - kind: invoke
        component: consume
        inputs: { image: request.image }
"#,
    )
    .expect("workflow metadata parses");
    let errors = validate_metadata(&metadata).expect_err("unguarded optional input must fail");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("outside the true case"))
    );
}

#[test]
fn request_presence_rejects_roles_with_implicit_defaults() {
    let metadata: InferenceMetadata = serde_yaml::from_str(
        r#"
schema_version: v1
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: { ai.onnx: 24 }
      capabilities: [workflow_ssa, input_presence]
    inputs:
      request.temperature:
        contract: { dtype: float32, rank: 1, shape: [1] }
        role: { kind: runtime, version: "1.0", role: sampling_temperature }
        source: { kind: request }
        required: false
        present_as: request.temperature_present
    components: {}
    steps: []
"#,
    )
    .expect("workflow metadata parses");
    let errors = validate_metadata(&metadata).expect_err("implicit request default must fail");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("whose absence is observable"))
    );
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
fn removed_legacy_kv_selection_surfaces_are_rejected() {
    for document in [
        "kv_cache: { native_dtype: float16 }\n",
        "model:\n  io:\n    kv_update: shared_buffer\n",
        "model:\n  runtime_configurable:\n    kv_cache: { dtype: [float16] }\n",
    ] {
        let error = serde_yaml::from_str::<InferenceMetadata>(document)
            .expect_err("legacy KV selection metadata must not deserialize");
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
fn row_emit_requires_explicit_int64_identities() {
    fn validate(
        row_ids: &str,
        ids_batch: &str,
        capabilities: &str,
        extra: &str,
    ) -> Result<(), Vec<String>> {
        let metadata: InferenceMetadata = serde_yaml::from_str(&format!(
            r#"
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: {{ ai.onnx: 24 }}
      capabilities: [workflow_ssa, typed_emit, emit_valid_length{capabilities}]
    inputs:
      value:
        contract: {{ dtype: int64, rank: 2, shape: [batch, sequence] }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: value }}
        required: true
      length:
        contract: {{ dtype: int64, rank: 1, shape: [batch] }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: length }}
        required: true
      ids:
        contract: {{ dtype: int64, rank: 1, shape: [{ids_batch}] }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: ids }}
        required: true
    outputs:
      result:
        contract: {{ dtype: int64, rank: 2, shape: [batch, generated] }}
        role: tokens
        stage: pre_adapter
    components: {{}}
    steps:
      - kind: emit
        value: value
        valid_length: length
        {row_ids}
        output: result
        mode: append
{extra}
"#
        ))
        .expect("workflow metadata parses");
        validate_metadata(&metadata)
    }

    let errors = validate("", "batch", "", "").expect_err("implicit row identity must fail");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("requires explicit row_ids"))
    );
    validate("row_ids: ids", "batch", ", emit_row_identity", "")
        .expect("explicit row identity is valid");
    let errors = validate("row_ids: ids", "1", ", emit_row_identity", "")
        .expect_err("row identity batch must match the emitted tensor");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("batch dimension must match"))
    );
    let errors = validate(
        "row_ids: ids",
        "batch",
        ", emit_row_identity",
        "      - kind: emit\n        value: value\n        output: result\n        mode: append",
    )
    .expect_err("one output cannot mix aggregate and row-wise emits");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("mixes aggregate and row-wise emits"))
    );
}

#[test]
fn serving_compaction_keeps_emit_identity_in_the_carried_slot_permutation() {
    let fixture = include_str!(
        "../../../tests/fixtures/onnx_genai_workflows/decoder/inference_metadata.yaml"
    );
    let compacted = fixture.to_owned();
    let mismatched = compacted.replacen("row_ids: slot_ids", "row_ids: active", 1);
    let metadata: InferenceMetadata =
        serde_yaml::from_str(&mismatched).expect("modified decoder metadata parses");
    let errors = validate_metadata(&metadata).expect_err("compacted identity must use slot_ids");
    assert!(
        errors
            .iter()
            .any(|error| { error.contains("row_ids must reference serving slot_ids 'slot_ids'") })
    );

    let uncarried = compacted.replacen("      - cell: slot_ids\n        next: slot_ids\n", "", 1);
    let metadata: InferenceMetadata =
        serde_yaml::from_str(&uncarried).expect("modified decoder metadata parses");
    let errors = validate_metadata(&metadata).expect_err("compacted slot IDs must be carried");
    assert!(
        errors
            .iter()
            .any(|error| { error.contains("carried must preserve serving slot_ids 'slot_ids'") })
    );

    let corrupted = compacted.replacen(
        "      - cell: slot_ids\n        next: slot_ids\n",
        "      - cell: slot_ids\n        next: cache_lengths.next\n",
        1,
    );
    let metadata: InferenceMetadata =
        serde_yaml::from_str(&corrupted).expect("modified decoder metadata parses");
    let errors =
        validate_metadata(&metadata).expect_err("compacted slot identity cannot change provenance");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("next value must be the carried slot_ids value"))
    );
}

#[test]
fn nested_control_loops_inherit_the_outer_compaction_permutation() {
    let metadata: InferenceMetadata = serde_yaml::from_str(
        r#"
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: { ai.onnx: 24 }
      capabilities:
        - workflow_ssa
        - linear_effects
        - nested_control_flow
        - typed_emit
        - emit_valid_length
        - emit_row_identity
        - serving_service_contract
    inputs:
      value:
        contract: { dtype: int64, rank: 2, shape: [batch, sequence] }
        role: { kind: opaque }
        source: { kind: application, name: value }
        required: true
      valid_length:
        contract: { dtype: int64, rank: 1, shape: [batch] }
        role: { kind: opaque }
        source: { kind: application, name: valid_length }
        required: true
      active.initial:
        contract: { dtype: bool, rank: 1, shape: [batch] }
        role: { kind: opaque }
        source: { kind: application, name: active }
        required: true
      done.initial:
        contract: { dtype: bool, rank: 1, shape: [batch] }
        role: { kind: opaque }
        source: { kind: application, name: done }
        required: true
      slot_ids.initial:
        contract: { dtype: int64, rank: 1, shape: [batch] }
        role: { kind: opaque }
        source: { kind: application, name: slot_ids }
        required: true
      cache.initial:
        contract: { dtype: float32, rank: 2, shape: [batch, capacity] }
        role: { kind: opaque }
        source: { kind: application, name: cache }
        required: true
      max_iterations:
        contract: { dtype: int64, rank: 0, shape: [] }
        role: { kind: opaque }
        source: { kind: application, name: max_iterations }
        required: true
    outputs:
      result:
        contract: { dtype: int64, rank: 2, shape: [batch, generated] }
        role: tokens
        stage: pre_adapter
    components: {}
    state:
      active:
        contract: { dtype: bool, rank: 1, shape: [batch] }
        scope: invocation
        initializer: active.initial
        recurrence: { kind: invariant }
      done:
        contract: { dtype: bool, rank: 1, shape: [batch] }
        scope: invocation
        initializer: done.initial
        recurrence: { kind: invariant }
      slot_ids:
        contract: { dtype: int64, rank: 1, shape: [batch] }
        scope: invocation
        initializer: slot_ids.initial
        recurrence: { kind: invariant }
      accepted_len:
        contract: { dtype: int64, rank: 1, shape: [batch] }
        scope: invocation
        initializer: valid_length
        recurrence: { kind: invariant }
      cache:
        contract: { dtype: float32, rank: 2, shape: [batch, capacity] }
        scope: invocation
        initializer: cache.initial
        recurrence: { kind: invariant }
        service_group: cache
    serving:
      active: active
      done: done
      accepted_len: accepted_len
      slot_ids: slot_ids
      kv_service:
        paging: none
        allocation: runtime
        compaction: true
        groups:
          cache:
            sequence_axis: 1
            layout: batch_sequence
            logical_lengths: accepted_len
            storage: shared_buffer
    steps:
      - kind: loop
        continue_when: active
        max_iterations: max_iterations
        carried:
          - { cell: active, next: active }
          - { cell: done, next: done }
          - { cell: slot_ids, next: slot_ids }
          - { cell: accepted_len, next: accepted_len }
          - { cell: cache, next: cache }
        steps:
          - kind: loop
            continue_when: active
            max_iterations: max_iterations
            steps:
              - kind: emit
                value: value
                valid_length: accepted_len
                row_ids: slot_ids
                output: result
                mode: append
"#,
    )
    .expect("nested compaction workflow parses");
    validate_metadata(&metadata)
        .expect("nested loops may use the invariant slot permutation carried by the outer loop");
    let graph = compile_workflow(&metadata.pipeline.expect("pipeline").workflow)
        .expect("nested compaction workflow lowers")
        .graph;
    let WorkflowNode::Sequence { nodes } = graph else {
        panic!("workflow must lower to a sequence");
    };
    let WorkflowNode::Loop { body, .. } = &nodes[0] else {
        panic!("outer lifecycle loop must be preserved");
    };
    let WorkflowNode::Sequence { nodes } = body.as_ref() else {
        panic!("outer loop body must be a sequence");
    };
    let WorkflowNode::Loop { body, carried, .. } = &nodes[0] else {
        panic!("nested control loop must be preserved");
    };
    assert!(carried.is_empty(), "nested loop must not redefine slot_ids");
    let WorkflowNode::Sequence { nodes } = body.as_ref() else {
        panic!("nested loop body must be a sequence");
    };
    let WorkflowNode::Emit { row_ids, .. } = &nodes[0] else {
        panic!("nested row-wise emit must be preserved");
    };
    assert_eq!(row_ids.as_deref(), Some("slot_ids"));
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

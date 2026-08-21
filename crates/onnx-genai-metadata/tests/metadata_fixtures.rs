use onnx_genai_metadata::{InferenceMetadata, WorkflowNode, compile_workflow, validate_metadata};

const ADAPTER_WORKFLOW: &str = r#"
schema_version: v1
adapters:
  target_manifest:
    targets:
      - id: projection
        component: decoder
        initializer: projection.weight
        layer_index: 0
        node_name: projection
        output_name: projection.output
        activation_dtype: float32
        input_features: 2
        output_features: 2
        rank: 1
        alpha: 1.0
        output_slice:
          role: projection
          offset: 0
          width: 2
          rank: 1
          alpha: 1.0
  selection:
    segments: request.adapter_segments
    adapter_counts: request.adapter_counts
    scales: request.adapter_scales
    max_adapters: 2
  application_capability: onnx-genai.adapters@1
  portable_fallback: true
  cache: { max_entries: 2, eviction: lru }
  planning:
    bucket_by_adapter_set: true
    stable_buffers: true
    invalidate_capture_on_eviction: true
  artifacts:
    red:
      index: 0
      identity: red
      version: "1"
      rank: 1
      alpha: 1.0
      dtype: float32
      provenance: { producer: synthetic-test }
      weights:
        - location: adapters/red/adapter.json
          loader_capability: onnx-genai.adapters.json@1
          scale_encoding: alpha_over_rank
          format: json
      bindings:
        - target: projection
          weight_key: projection
pipeline:
  workflow:
    manifest:
      adapter_abis: { onnx-genai.parameter-overlay: "1" }
      capabilities: [workflow_ssa, parameter_adapters, heterogeneous_adapter_batching]
    inputs:
      request.adapter_segments:
        contract: { dtype: int64, rank: 2, shape: [batch, 2], batch_layout: { kind: request_aligned, axis: 0 } }
        role: { kind: runtime, version: "1.0", role: adapter_segments }
        source: { kind: request }
      request.adapter_counts:
        contract: { dtype: int64, rank: 1, shape: [batch], batch_layout: { kind: request_aligned, axis: 0 } }
        role: { kind: runtime, version: "1.0", role: adapter_counts }
        source: { kind: request }
      request.adapter_scales:
        contract: { dtype: float32, rank: 2, shape: [batch, 2], batch_layout: { kind: request_aligned, axis: 0 } }
        role: { kind: runtime, version: "1.0", role: adapter_scales }
        source: { kind: request }
    components:
      decoder:
        implementation: { kind: binding }
        ports: {}
      overlay:
        implementation:
          kind: adapter
          abi: onnx-genai.parameter-overlay
          version: "1"
        ports:
          inputs:
            input: { dtype: float32, rank: 2, shape: [batch, 2] }
          outputs:
            output: { dtype: float32, rank: 2, shape: [batch, 2] }
        contract:
          id: onnx-genai.parameter-overlay
          version: "1"
          bindings: { input: input, output: output }
          parameters:
            action: apply
            component: decoder
            parameter: projection.weight
    steps:
      - kind: invoke
        component: decoder
"#;

#[test]
fn adapter_service_contract_is_valid_and_derives_capabilities() {
    let metadata: InferenceMetadata =
        serde_yaml::from_str(ADAPTER_WORKFLOW).expect("adapter workflow parses");
    validate_metadata(&metadata).expect("adapter workflow validates");
    let capabilities = onnx_genai_metadata::derived_capabilities(&metadata);
    assert!(capabilities.contains("parameter_adapters"));
    assert!(capabilities.contains("heterogeneous_adapter_batching"));
}

#[test]
fn workflow_custom_op_admission_fields_are_rejected() {
    let manifest_field = ADAPTER_WORKFLOW.replace(
        "adapter_abis: { onnx-genai.parameter-overlay: \"1\" }",
        "adapter_abis: { onnx-genai.parameter-overlay: \"1\" }\n      \
         custom_op_versions: { com.example: \"1\" }",
    );
    let error = serde_yaml::from_str::<InferenceMetadata>(&manifest_field)
        .expect_err("custom-op manifest field must be rejected");
    assert!(error.to_string().contains("custom_op_versions"));

    let component_field = ADAPTER_WORKFLOW.replace(
        "version: \"1\"\n        ports:",
        "version: \"1\"\n          custom_ops: { com.example: \"1\" }\n        ports:",
    );
    let error = serde_yaml::from_str::<InferenceMetadata>(&component_field)
        .expect_err("custom-op component field must be rejected");
    assert!(error.to_string().contains("custom_ops"));
}

#[test]
fn adapter_service_rejects_incompatible_or_unsafe_artifacts() {
    let invalid = ADAPTER_WORKFLOW.replace(
        "location: adapters/red/adapter.json",
        "location: ../outside/red.json",
    );
    let metadata: InferenceMetadata =
        serde_yaml::from_str(&invalid).expect("invalid adapter workflow parses");
    let errors = validate_metadata(&metadata).expect_err("invalid adapter workflow must fail");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("must be under package path adapters/red/"))
    );
}

#[test]
fn retired_artifact_hash_fields_are_rejected_as_unknown() {
    for (name, document, field) in [
        (
            "component",
            r#"
pipeline:
  workflow:
    components:
      model:
        implementation: {kind: onnx, artifact: model.onnx, sha256: retired}
        ports: {}
    steps: []
"#
            .to_string(),
            "sha256",
        ),
        (
            "tokenizer",
            r#"
package:
  tokenizer:
    algorithm: bpe
    vocab_size: 1
    artifacts:
      - {location: tokenizer.json, sha256: retired}
"#
            .to_string(),
            "sha256",
        ),
        (
            "adapter service",
            ADAPTER_WORKFLOW.replace(
                "adapters:\n",
                "adapters:\n  base_model_fingerprint: retired\n",
            ),
            "base_model_fingerprint",
        ),
        (
            "adapter artifact",
            ADAPTER_WORKFLOW.replace(
                "          loader_capability: onnx-genai.adapters.json@1\n",
                "          loader_capability: onnx-genai.adapters.json@1\n          sha256: retired\n",
            ),
            "sha256",
        ),
        (
            "adapter config",
            ADAPTER_WORKFLOW.replace(
                "          loader_capability: onnx-genai.adapters.json@1\n",
                "          loader_capability: onnx-genai.adapters.json@1\n          config_sha256: retired\n",
            ),
            "config_sha256",
        ),
    ] {
        let error = serde_yaml::from_str::<InferenceMetadata>(&document)
            .expect_err(&format!("retired {name} field must be rejected"));
        assert!(error.to_string().contains(field), "{name}: {error}");
    }
}

#[test]
fn adapter_wire_contract_rejects_ambiguous_selection_and_indices() {
    let invalid = ADAPTER_WORKFLOW
        .replace("max_adapters: 2", "max_adapters: 0")
        .replace("index: 0", "index: 2")
        .replace(
            "contract: { dtype: int64, rank: 2, shape: [batch, 2] }\n        role: { kind: runtime, version: \"1.0\", role: adapter_segments }",
            "contract: { dtype: float32, rank: 2, shape: [batch, 2] }\n        role: { kind: runtime, version: \"1.0\", role: adapter_segments }",
        );
    let metadata: InferenceMetadata =
        serde_yaml::from_str(&invalid).expect("invalid adapter wire contract parses");
    let errors = validate_metadata(&metadata).expect_err("ambiguous adapter wire must fail");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("max_adapters must be greater than zero"))
    );
    assert!(
        errors
            .iter()
            .any(|error| error.contains("artifact indices must be contiguous from zero"))
    );
    assert!(
        errors
            .iter()
            .any(|error| error.contains("selection.segments"))
    );
}

#[test]
fn adapter_manifest_and_source_contracts_fail_loud() {
    let invalid = ADAPTER_WORKFLOW
        .replace("target: projection", "target: missing")
        .replace(
            "loader_capability: onnx-genai.adapters.json@1",
            "loader_capability: onnxruntime.lora-adapter@1",
        )
        .replace("scale_encoding: alpha_over_rank", "scale_encoding: baked")
        .replace("format: json", "format: hf_peft");
    let metadata: InferenceMetadata =
        serde_yaml::from_str(&invalid).expect("invalid adapter contracts parse");
    let errors = validate_metadata(&metadata).expect_err("invalid adapter contracts must fail");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("absent from adapters.target_manifest"))
    );
    assert!(
        errors
            .iter()
            .any(|error| error.contains("loader_capability must be"))
    );
    assert!(
        errors
            .iter()
            .any(|error| error.contains("hf_peft requires config_location"))
    );
    assert!(
        errors
            .iter()
            .any(|error| error.contains("scale_encoding must be alpha_over_rank"))
    );
    let policy_invalid = ADAPTER_WORKFLOW.replacen("        rank: 1", "        rank: 2", 1);
    let metadata: InferenceMetadata =
        serde_yaml::from_str(&policy_invalid).expect("invalid target policy parses");
    let errors = validate_metadata(&metadata).expect_err("target policy mismatch must fail");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("violates target policy"))
    );
    let slice_policy_invalid =
        ADAPTER_WORKFLOW.replacen("          rank: 1", "          rank: 2", 1);
    let metadata: InferenceMetadata =
        serde_yaml::from_str(&slice_policy_invalid).expect("invalid slice policy parses");
    let errors = validate_metadata(&metadata).expect_err("output-slice policy mismatch must fail");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("violates output-slice policy"))
    );
}

#[test]
fn minimal_workflow_document_is_valid() {
    let metadata: InferenceMetadata = serde_yaml::from_str(
        r#"
schema_version: v1
pipeline:
  workflow:
    manifest:
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
        "tokens",
        "speculator_config",
    ] {
        let document = format!("{field}: {{}}\n");
        let error = serde_yaml::from_str::<InferenceMetadata>(&document)
            .expect_err("removed top-level execution metadata must not deserialize");
        assert!(error.to_string().contains("unknown field"), "{error}");
    }

    // `generation` and `speculative` exist again, but as typed declarations of
    // facts, not as runtime knobs. The legacy free-form shapes still fail.
    for (document, expected) in [
        ("generation:\n  do_sample: true\n", "unknown field"),
        (
            "speculative:\n  num_speculative_tokens: 4\n",
            "unknown field",
        ),
        ("speculative: {}\n", "missing field"),
    ] {
        let error = serde_yaml::from_str::<InferenceMetadata>(document)
            .expect_err("legacy execution knobs must not deserialize");
        assert!(
            error.to_string().contains(expected),
            "expected {expected}: {error}"
        );
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
fn row_wise_emit_requires_a_request_aligned_batch_layout() {
    fn validate(batch_layout: &str, extra: &str) -> Result<(), Vec<String>> {
        let metadata: InferenceMetadata = serde_yaml::from_str(&format!(
            r#"
pipeline:
  workflow:
    manifest:
      capabilities: [workflow_ssa, typed_emit, emit_valid_length]
    inputs:
      value:
        contract:
          dtype: int64
          rank: 2
          shape: [batch, sequence]
          {batch_layout}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: value }}
        required: true
      length:
        contract: {{ dtype: int64, rank: 1, shape: [batch] }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: length }}
        required: true
    outputs:
      result:
        contract:
          dtype: int64
          rank: 2
          shape: [batch, generated]
          {batch_layout}
        role: tokens
        stage: pre_adapter
    components: {{}}
    steps:
      - kind: emit
        value: value
        valid_length: length
        output: result
        mode: append
{extra}
"#
        ))
        .expect("workflow metadata parses");
        validate_metadata(&metadata)
    }

    // A row-wise prefix length has no meaning without a request-aligned axis:
    // the runtime would have no derivable way to attach result rows to requests.
    let errors =
        validate("", "").expect_err("row-wise emit without a request axis must fail closed");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("declares no request_aligned batch_layout")),
        "{errors:?}"
    );

    validate("batch_layout: { kind: request_aligned, axis: 0 }", "")
        .expect("a request-aligned emitted value is valid");

    // Two emits into one output must agree, or the runtime cannot tell whether
    // the accumulated output is row-scoped.
    let errors = validate(
        "batch_layout: { kind: request_aligned, axis: 0 }",
        "      - kind: emit\n        value: length\n        output: result\n        mode: append",
    )
    .expect_err("one output cannot mix batch layouts across emits");
    assert!(
        errors.iter().any(|error| {
            error.contains("mixes batch layouts across emits")
                || error.contains("incompatible dtype or rank")
        }),
        "{errors:?}"
    );
}

#[test]
fn removed_row_identity_fields_are_rejected_fail_closed() {
    // Row identity is never serialized. A package that still ships the retired
    // fields must fail to parse rather than have them silently ignored.
    for retired in [
        "      - kind: emit\n        value: value\n        row_ids: ids\n        output: result\n        mode: append",
        "      - kind: emit\n        value: value\n        output: result\n        mode: append\n        row_ids: ids",
    ] {
        let document = format!(
            r#"
pipeline:
  workflow:
    manifest:
      capabilities: [workflow_ssa]
    inputs:
      value:
        contract: {{ dtype: int64, rank: 2, shape: [batch, sequence] }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: value }}
        required: true
    outputs:
      result:
        contract: {{ dtype: int64, rank: 2, shape: [batch, generated] }}
        role: tokens
        stage: pre_adapter
    components: {{}}
    steps:
{retired}
"#
        );
        let parsed = serde_yaml::from_str::<InferenceMetadata>(&document);
        assert!(
            parsed.is_err(),
            "retired emit.row_ids must be rejected, not ignored"
        );
    }

    // The retired serving and adapter row-identity fields are likewise closed.
    for retired in [
        "    serving:\n      slot_ids: slot_ids\n",
        "    adapters:\n      selection:\n        slot_ids: slot_ids\n",
    ] {
        let document = format!(
            r#"
pipeline:
  workflow:
    manifest:
      capabilities: [workflow_ssa]
    inputs: {{}}
    outputs: {{}}
    components: {{}}
    steps: []
{retired}
"#
        );
        assert!(
            serde_yaml::from_str::<InferenceMetadata>(&document).is_err(),
            "retired row-identity field must be rejected: {retired}"
        );
    }
}

#[test]
fn nested_control_loops_preserve_the_request_aligned_emit() {
    let metadata: InferenceMetadata = serde_yaml::from_str(
        r#"
pipeline:
  workflow:
    manifest:
      capabilities:
        - workflow_ssa
        - linear_effects
        - nested_control_flow
        - typed_emit
        - emit_valid_length
        - serving_service_contract
    inputs:
      value:
        contract:
          dtype: int64
          rank: 2
          shape: [batch, sequence]
          batch_layout: { kind: request_aligned, axis: 0 }
        role: { kind: opaque }
        source: { kind: application, name: value }
        required: true
      valid_length:
        contract:
          dtype: int64
          rank: 1
          shape: [batch]
          batch_layout: { kind: request_aligned, axis: 0 }
        role: { kind: opaque }
        source: { kind: application, name: valid_length }
        required: true
      active.initial:
        contract:
          dtype: bool
          rank: 1
          shape: [batch]
          batch_layout: { kind: request_aligned, axis: 0 }
        role: { kind: opaque }
        source: { kind: application, name: active }
        required: true
      done.initial:
        contract:
          dtype: bool
          rank: 1
          shape: [batch]
          batch_layout: { kind: request_aligned, axis: 0 }
        role: { kind: opaque }
        source: { kind: application, name: done }
        required: true
      cache.initial:
        contract:
          dtype: float32
          rank: 2
          shape: [batch, capacity]
          batch_layout: { kind: request_aligned, axis: 0 }
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
        contract:
          dtype: int64
          rank: 2
          shape: [batch, generated]
          batch_layout: { kind: request_aligned, axis: 0 }
        role: tokens
        stage: pre_adapter
    components: {}
    state:
      active:
        contract:
          dtype: bool
          rank: 1
          shape: [batch]
          batch_layout: { kind: request_aligned, axis: 0 }
        scope: invocation
        initializer: active.initial
        recurrence: { kind: invariant }
      done:
        contract:
          dtype: bool
          rank: 1
          shape: [batch]
          batch_layout: { kind: request_aligned, axis: 0 }
        scope: invocation
        initializer: done.initial
        recurrence: { kind: invariant }
      accepted_len:
        contract:
          dtype: int64
          rank: 1
          shape: [batch]
          batch_layout: { kind: request_aligned, axis: 0 }
        scope: invocation
        initializer: valid_length
        recurrence: { kind: invariant }
      cache:
        contract:
          dtype: float32
          rank: 2
          shape: [batch, capacity]
          batch_layout: { kind: request_aligned, axis: 0 }
        scope: invocation
        initializer: cache.initial
        recurrence: { kind: invariant }
        service_group: cache
        management: runtime
        release_boundary: invocation
    serving:
      active: active
      done: done
      accepted_len: accepted_len
      state_service:
        groups:
          cache:
            kind: full_attention
            sequence_axis: 1
            layout: batch_sequence
            logical_lengths: accepted_len
            aliasing: permitted
            capabilities:
              rollback_positions: 8
    steps:
      - kind: loop
        continue_when: active
        max_iterations: max_iterations
        carried:
          - { cell: active, next: active }
          - { cell: done, next: done }
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
                output: result
                mode: append
"#,
    )
    .expect("nested workflow parses");
    validate_metadata(&metadata).expect("nested loops may emit request-aligned rows");
    let graph = compile_workflow(&metadata.pipeline.expect("pipeline").workflow)
        .expect("nested workflow lowers")
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
    assert!(carried.is_empty(), "nested loop must not redefine state");
    let WorkflowNode::Sequence { nodes } = body.as_ref() else {
        panic!("nested loop body must be a sequence");
    };
    let WorkflowNode::Emit { value, output, .. } = &nodes[0] else {
        panic!("nested row-wise emit must be preserved");
    };
    assert_eq!(value, "value");
    assert_eq!(output, "result");
}

#[test]
fn advisory_state_may_be_session_scoped_but_is_not_semantic() {
    let metadata: InferenceMetadata = serde_yaml::from_str(
        r#"
pipeline:
  workflow:
    manifest:
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
fn state_service_declares_semantics_not_allocator_policy() {
    let metadata: InferenceMetadata = serde_yaml::from_str(
        r#"
pipeline:
  workflow:
    manifest:
      capabilities: [workflow_ssa, serving_service_contract]
    inputs:
      active:
        contract: { dtype: bool, rank: 1, shape: [batch], batch_layout: { kind: request_aligned, axis: 0 } }
        role: { kind: opaque }
        source: { kind: application, name: active }
      done:
        contract: { dtype: bool, rank: 1, shape: [batch], batch_layout: { kind: request_aligned, axis: 0 } }
        role: { kind: opaque }
        source: { kind: application, name: done }
      accepted_len:
        contract: { dtype: int64, rank: 1, shape: [batch], batch_layout: { kind: request_aligned, axis: 0 } }
        role: { kind: opaque }
        source: { kind: application, name: accepted_len }
      cache_lengths:
        contract: { dtype: int64, rank: 1, shape: [batch], batch_layout: { kind: request_aligned, axis: 0 } }
        role: { kind: opaque }
        source: { kind: application, name: cache_lengths }
      empty_cache:
        contract: { dtype: float16, rank: 4, shape: [batch, heads, sequence, head_dim], batch_layout: { kind: request_aligned, axis: 0 } }
        role: { kind: opaque }
        source: { kind: application, name: empty_cache }
    components:
      decoder:
        implementation: { kind: onnx, artifact: decoder.onnx }
        ports: { roles: { input_ids: token_ids } }
    state:
      cache:
        contract: { dtype: float16, rank: 4, shape: [batch, heads, sequence, head_dim], batch_layout: { kind: request_aligned, axis: 0 } }
        class: semantic
        scope: invocation
        initializer: empty_cache
        recurrence: { kind: invariant }
        service_group: decoder_cache
        management: runtime
        release_boundary: invocation
      cache_lengths:
        contract: { dtype: int64, rank: 1, shape: [batch], batch_layout: { kind: request_aligned, axis: 0 } }
        class: semantic
        scope: invocation
        initializer: cache_lengths
        recurrence: { kind: invariant }
    serving:
      active: active
      done: done
      accepted_len: accepted_len
      state_service:
        groups:
          decoder_cache:
            kind: full_attention
            sequence_axis: 2
            layout: bnsh
            logical_lengths: cache_lengths
            aliasing: permitted
            reuse: { prefix_reusable: true, evictable_prefix: true }
            capabilities: { rollback_positions: 32, snapshot: true, fork: true }
            ports:
              decoder:
                cache: { input: past_key_values, output: present_key_values }
    steps:
      - kind: invoke
        component: decoder
"#,
    )
    .expect("state service metadata parses");
    validate_metadata(&metadata).expect("state group is executable and bound");

    // The retired allocator-policy keys are runtime-owned and must not reappear.
    for retired in [
        "        paging: paged\n",
        "        allocation: runtime\n",
        "            storage: shared_buffer\n",
        "            compaction: slot_permutation\n",
    ] {
        let document = serde_yaml::to_string(
            &serde_yaml::from_str::<serde_yaml::Value>(
                "pipeline:\n  workflow:\n    serving:\n      state_service:\n        groups: {}\n",
            )
            .expect("skeleton"),
        )
        .expect("skeleton text");
        let injected = document.replace(
            "        groups: {}",
            &format!("{retired}        groups: {{}}"),
        );
        assert!(
            serde_yaml::from_str::<InferenceMetadata>(&injected).is_err(),
            "retired allocator policy must be rejected: {retired}"
        );
    }

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

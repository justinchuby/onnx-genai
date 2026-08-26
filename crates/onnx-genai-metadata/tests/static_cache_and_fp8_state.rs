//! What a fixed-capacity state group and an FP8 cache may declare.
//!
//! Two facts are pinned here, and they fail in opposite directions.
//!
//! A static cache separates capacity from length, so its declaration has parts a
//! growing cache does not need: a write cursor, a capacity, and a port to carry
//! destinations to the graph. Each of those is load-bearing, so each has a test
//! that removes it and expects a refusal — a declaration that is optional in
//! practice is not a contract.
//!
//! FP8 fails the other way. Its dtypes are already in the vocabulary, so the
//! risk is not that a package can say too little but that the validator says no
//! to something it has no business judging. Whether an execution provider has an
//! FP8 kernel is a runtime capability question; a schema that pre-empts it turns
//! a precise "this EP cannot do that" into a false "this document is invalid",
//! and the difference matters because only one of them tells you to change EP.

use onnx_genai_metadata::{
    InferenceMetadata, TensorDimension, semantic_identity_of_str, validate_metadata,
};

fn parse(document: &str) -> InferenceMetadata {
    serde_yaml::from_str(document).expect("metadata parses")
}

fn errors(document: &str) -> Vec<String> {
    validate_metadata(&parse(document)).expect_err("metadata must be rejected")
}

/// A minimal decoder whose cache is a fixed-capacity buffer written by scatter.
///
/// `DTYPE` is substituted so the same document can be checked in float32 and in
/// FP8: nothing about the contract changes with the element type, which is the
/// point of the FP8 tests below.
const STATIC_CACHE_WORKFLOW: &str = r#"
schema_version: v1
pipeline:
  workflow:
    manifest:
      capabilities:
      - workflow_ssa
      - linear_effects
      - typed_emit
      - nested_control_flow
      - loop_induction_values
      - serving_service_contract
    inputs:
      request.input_ids:
        contract:
          dtype: int64
          rank: 2
          shape: [batch, prompt_sequence]
          batch_layout: {kind: request_aligned, axis: 0}
        role: {kind: runtime, version: '1.0', role: prompt_tokens}
        source: {kind: request}
        required: true
      request.max_iterations:
        contract: {dtype: int64, rank: 1, shape: [1]}
        role: {kind: runtime, version: '1.0', role: max_iterations}
        source: {kind: request}
        required: false
        default: 4
      package.capacity:
        contract: {dtype: int64, rank: 1, shape: [1]}
        role: {kind: opaque}
        source: {kind: literal}
        required: false
        default: 8
      package.active:
        contract:
          dtype: bool
          rank: 1
          shape: [batch]
          batch_layout: {kind: request_aligned, axis: 0}
        role: {kind: opaque}
        source: {kind: literal}
        required: false
        default: true
      package.done:
        contract:
          dtype: bool
          rank: 1
          shape: [batch]
          batch_layout: {kind: request_aligned, axis: 0}
        role: {kind: opaque}
        source: {kind: literal}
        required: false
        default: false
      package.accepted_len:
        contract:
          dtype: int64
          rank: 1
          shape: [batch]
          batch_layout: {kind: request_aligned, axis: 0}
        role: {kind: opaque}
        source: {kind: literal}
        required: false
        default: 0
      package.loop_active:
        contract: {dtype: bool, rank: 1, shape: [1]}
        role: {kind: opaque}
        source: {kind: literal}
        required: false
        default: true
      package.write_indices:
        contract:
          dtype: int64
          rank: 1
          shape: [batch]
          batch_layout: {kind: request_aligned, axis: 0}
        role: {kind: opaque}
        source: {kind: literal}
        required: false
        default: 0
      package.cache_lengths:
        contract:
          dtype: int64
          rank: 1
          shape: [batch]
          batch_layout: {kind: request_aligned, axis: 0}
        role: {kind: opaque}
        source: {kind: literal}
        required: false
        default: 0
      package.cache:
        contract:
          dtype: DTYPE
          rank: 3
          shape: [batch, cache_capacity, 4]
          batch_layout: {kind: request_aligned, axis: 0}
        role: {kind: opaque}
        source: {kind: literal}
        required: false
        default: 0.0
    outputs:
      tokens:
        contract:
          dtype: int64
          rank: 2
          shape: [batch, generated]
          batch_layout: {kind: request_aligned, axis: 0}
        role: tokens
        stage: pre_adapter
    components:
      model:
        implementation: {kind: onnx, artifact: model.onnx}
        ports:
          roles:
            input_ids: token_ids
          inputs:
            input_ids:
              dtype: int64
              rank: 2
              shape: [batch, 1]
              batch_layout: {kind: request_aligned, axis: 0}
            cache:
              dtype: DTYPE
              rank: 3
              shape: [batch, cache_capacity, 4]
              batch_layout: {kind: request_aligned, axis: 0}
            write_indices:
              dtype: int64
              rank: 1
              shape: [batch]
              batch_layout: {kind: request_aligned, axis: 0}
          outputs:
            token:
              dtype: int64
              rank: 2
              shape: [batch, 1]
              batch_layout: {kind: request_aligned, axis: 0}
            updated_cache:
              dtype: DTYPE
              rank: 3
              shape: [batch, cache_capacity, 4]
              batch_layout: {kind: request_aligned, axis: 0}
            next_write_indices:
              dtype: int64
              rank: 1
              shape: [batch]
              batch_layout: {kind: request_aligned, axis: 0}
            next_cache_lengths:
              dtype: int64
              rank: 1
              shape: [batch]
              batch_layout: {kind: request_aligned, axis: 0}
            next_active:
              dtype: bool
              rank: 1
              shape: [batch]
              batch_layout: {kind: request_aligned, axis: 0}
            next_done:
              dtype: bool
              rank: 1
              shape: [batch]
              batch_layout: {kind: request_aligned, axis: 0}
            next_accepted_len:
              dtype: int64
              rank: 1
              shape: [batch]
              batch_layout: {kind: request_aligned, axis: 0}
            next_loop_active:
              dtype: bool
              rank: 1
              shape: [1]
    state:
      token:
        contract:
          dtype: int64
          rank: 2
          shape: [batch, 1]
          batch_layout: {kind: request_aligned, axis: 0}
        scope: invocation
        initializer: request.input_ids
        recurrence: {kind: invariant}
      active:
        contract:
          dtype: bool
          rank: 1
          shape: [batch]
          batch_layout: {kind: request_aligned, axis: 0}
        class: semantic
        scope: invocation
        initializer: package.active
        recurrence: {kind: invariant}
      done:
        contract:
          dtype: bool
          rank: 1
          shape: [batch]
          batch_layout: {kind: request_aligned, axis: 0}
        class: semantic
        scope: invocation
        initializer: package.done
        recurrence: {kind: invariant}
      accepted_len:
        contract:
          dtype: int64
          rank: 1
          shape: [batch]
          batch_layout: {kind: request_aligned, axis: 0}
        class: semantic
        scope: invocation
        initializer: package.accepted_len
        recurrence: {kind: invariant}
      loop_active:
        contract: {dtype: bool, rank: 1, shape: [1]}
        scope: invocation
        initializer: package.loop_active
        recurrence: {kind: invariant}
      write_indices:
        contract:
          dtype: int64
          rank: 1
          shape: [batch]
          batch_layout: {kind: request_aligned, axis: 0}
        class: semantic
        scope: invocation
        initializer: package.write_indices
        recurrence: {kind: invariant}
      cache_lengths:
        contract:
          dtype: int64
          rank: 1
          shape: [batch]
          batch_layout: {kind: request_aligned, axis: 0}
        class: semantic
        scope: invocation
        initializer: package.cache_lengths
        recurrence: {kind: invariant}
      cache:
        contract:
          dtype: DTYPE
          rank: 3
          shape: [batch, cache_capacity, 4]
          batch_layout: {kind: request_aligned, axis: 0}
        scope: invocation
        initializer: package.cache
        recurrence: {kind: invariant}
        service_group: decoder_cache
        management: runtime
        release_boundary: invocation
    serving:
      active: active
      done: done
      accepted_len: accepted_len
      state_service:
        groups:
          decoder_cache:
            kind: full_attention
            sequence_axis: 1
            layout: bsh
            logical_lengths: cache_lengths
            aliasing: permitted
            update:
              kind: indexed_scatter
              write_indices: write_indices
              capacity: package.capacity
              write_indices_ports:
                model: write_indices
            reuse: {prefix_reusable: true, evictable_prefix: true}
            capabilities: {rollback_positions: 8, snapshot: true, fork: true}
            ports:
              model:
                cache: {input: cache, output: updated_cache}
    steps:
    - kind: loop
      setup: []
      steps:
      - kind: invoke
        component: model
        inputs:
          input_ids: token
          cache: cache
          write_indices: write_indices
        outputs:
          token: body.token
          updated_cache: body.cache
          next_write_indices: body.write_indices
          next_cache_lengths: body.cache_lengths
          next_active: body.active
          next_done: body.done
          next_accepted_len: body.accepted_len
          next_loop_active: body.loop_active
      - kind: emit
        value: body.token
        output: tokens
        mode: append
        when: active
      continue_when: loop_active
      max_iterations: request.max_iterations
      carried:
      - {cell: token, next: body.token}
      - {cell: cache, next: body.cache}
      - {cell: write_indices, next: body.write_indices}
      - {cell: cache_lengths, next: body.cache_lengths}
      - {cell: active, next: body.active}
      - {cell: done, next: body.done}
      - {cell: accepted_len, next: body.accepted_len}
      - {cell: loop_active, next: body.loop_active}
      iteration:
        value: loop.iteration
        contract: {dtype: int64, rank: 1, shape: [1]}
package:
  tokenizer:
    algorithm: bpe
    vocab_size: 32
    byte_level: true
    artifacts:
    - location: tokenizer.json
"#;

/// The shape a real exporter emits: a port ABI beside the workflow that binds it.
///
/// Modeled on the Mobius qwen2 static-cache export, including its design choice
/// to use one cell for both the write cursor and the valid length. That is
/// coherent -- a row's next write lands exactly where its valid prefix ends --
/// and it is strictly safer than carrying two cells, because a row that stops
/// advancing parks on one slot instead of walking past capacity.
const MOBIUS_STATIC_CACHE: &str = r#"
schema_version: v1
pipeline:
  workflow:
    manifest:
      capabilities:
      - workflow_ssa
      - linear_effects
      - typed_emit
      - nested_control_flow
      - loop_induction_values
      - serving_service_contract
    inputs:
      request.input_ids:
        contract:
          dtype: int64
          rank: 2
          shape: [batch, prompt_sequence]
          batch_layout: {kind: request_aligned, axis: 0}
        role: {kind: runtime, version: '1.0', role: prompt_tokens}
        source: {kind: request}
        required: true
      request.max_iterations:
        contract: {dtype: int64, rank: 1, shape: [1]}
        role: {kind: runtime, version: '1.0', role: max_iterations}
        source: {kind: request}
        required: false
        default: 4
      package.capacity:
        contract: {dtype: int64, rank: 1, shape: [1]}
        role: {kind: opaque}
        source: {kind: literal}
        required: false
        default: 8
      package.active:
        contract:
          dtype: bool
          rank: 1
          shape: [batch]
          batch_layout: {kind: request_aligned, axis: 0}
        role: {kind: opaque}
        source: {kind: literal}
        required: false
        default: true
      package.done:
        contract:
          dtype: bool
          rank: 1
          shape: [batch]
          batch_layout: {kind: request_aligned, axis: 0}
        role: {kind: opaque}
        source: {kind: literal}
        required: false
        default: false
      package.accepted_len:
        contract:
          dtype: int64
          rank: 1
          shape: [batch]
          batch_layout: {kind: request_aligned, axis: 0}
        role: {kind: opaque}
        source: {kind: literal}
        required: false
        default: 0
      package.loop_active:
        contract: {dtype: bool, rank: 1, shape: [1]}
        role: {kind: opaque}
        source: {kind: literal}
        required: false
        default: true
      package.write_indices:
        contract:
          dtype: int64
          rank: 1
          shape: [batch]
          batch_layout: {kind: request_aligned, axis: 0}
        role: {kind: opaque}
        source: {kind: literal}
        required: false
        default: 0
      package.cache_lengths:
        contract:
          dtype: int64
          rank: 1
          shape: [batch]
          batch_layout: {kind: request_aligned, axis: 0}
        role: {kind: opaque}
        source: {kind: literal}
        required: false
        default: 0
      package.cache:
        contract:
          dtype: DTYPE
          rank: 3
          shape: [batch, cache_capacity, 4]
          batch_layout: {kind: request_aligned, axis: 0}
        role: {kind: opaque}
        source: {kind: literal}
        required: false
        default: 0.0
    outputs:
      tokens:
        contract:
          dtype: int64
          rank: 2
          shape: [batch, generated]
          batch_layout: {kind: request_aligned, axis: 0}
        role: tokens
        stage: pre_adapter
    components:
      model:
        implementation: {kind: onnx, artifact: model.onnx}
        ports:
          inputs:
            input_ids:
              dtype: int64
              rank: 2
              shape: [batch, 1]
              batch_layout: {kind: request_aligned, axis: 0}
            key_cache.0:
              dtype: DTYPE
              rank: 3
              shape: [batch, cache_capacity, 4]
              batch_layout: {kind: request_aligned, axis: 0}
            value_cache.0:
              dtype: DTYPE
              rank: 3
              shape: [batch, cache_capacity, 4]
              batch_layout: {kind: request_aligned, axis: 0}
            write_indices:
              dtype: int64
              rank: 1
              shape: [batch]
              batch_layout: {kind: request_aligned, axis: 0}
            nonpad_kv_seqlen:
              dtype: int64
              rank: 1
              shape: [batch]
              batch_layout: {kind: request_aligned, axis: 0}
          outputs:
            token:
              dtype: int64
              rank: 2
              shape: [batch, 1]
              batch_layout: {kind: request_aligned, axis: 0}
            updated_key_cache.0:
              dtype: DTYPE
              rank: 3
              shape: [batch, cache_capacity, 4]
              batch_layout: {kind: request_aligned, axis: 0}
            updated_value_cache.0:
              dtype: DTYPE
              rank: 3
              shape: [batch, cache_capacity, 4]
              batch_layout: {kind: request_aligned, axis: 0}
            next_write_indices:
              dtype: int64
              rank: 1
              shape: [batch]
              batch_layout: {kind: request_aligned, axis: 0}
            next_cache_lengths:
              dtype: int64
              rank: 1
              shape: [batch]
              batch_layout: {kind: request_aligned, axis: 0}
            next_active:
              dtype: bool
              rank: 1
              shape: [batch]
              batch_layout: {kind: request_aligned, axis: 0}
            next_done:
              dtype: bool
              rank: 1
              shape: [batch]
              batch_layout: {kind: request_aligned, axis: 0}
            next_accepted_len:
              dtype: int64
              rank: 1
              shape: [batch]
              batch_layout: {kind: request_aligned, axis: 0}
            next_loop_active:
              dtype: bool
              rank: 1
              shape: [1]
          roles:
            input_ids: token_ids
    state:
      token:
        contract:
          dtype: int64
          rank: 2
          shape: [batch, 1]
          batch_layout: {kind: request_aligned, axis: 0}
        scope: invocation
        initializer: request.input_ids
        recurrence: {kind: invariant}
      active:
        contract:
          dtype: bool
          rank: 1
          shape: [batch]
          batch_layout: {kind: request_aligned, axis: 0}
        class: semantic
        scope: invocation
        initializer: package.active
        recurrence: {kind: invariant}
      done:
        contract:
          dtype: bool
          rank: 1
          shape: [batch]
          batch_layout: {kind: request_aligned, axis: 0}
        class: semantic
        scope: invocation
        initializer: package.done
        recurrence: {kind: invariant}
      accepted_len:
        contract:
          dtype: int64
          rank: 1
          shape: [batch]
          batch_layout: {kind: request_aligned, axis: 0}
        class: semantic
        scope: invocation
        initializer: package.accepted_len
        recurrence: {kind: invariant}
      loop_active:
        contract: {dtype: bool, rank: 1, shape: [1]}
        scope: invocation
        initializer: package.loop_active
        recurrence: {kind: invariant}
      write_indices:
        contract:
          dtype: int64
          rank: 1
          shape: [batch]
          batch_layout: {kind: request_aligned, axis: 0}
        class: semantic
        scope: invocation
        initializer: package.write_indices
        recurrence: {kind: invariant}
      cache_lengths:
        contract:
          dtype: int64
          rank: 1
          shape: [batch]
          batch_layout: {kind: request_aligned, axis: 0}
        class: semantic
        scope: invocation
        initializer: package.cache_lengths
        recurrence: {kind: invariant}
      key_cache:
        contract:
          dtype: DTYPE
          rank: 3
          shape: [batch, cache_capacity, 4]
          batch_layout: {kind: request_aligned, axis: 0}
        scope: invocation
        initializer: package.cache
        recurrence: {kind: invariant}
        service_group: decoder_cache
        management: runtime
        release_boundary: invocation
      value_cache:
        contract:
          dtype: DTYPE
          rank: 3
          shape: [batch, cache_capacity, 4]
          batch_layout: {kind: request_aligned, axis: 0}
        scope: invocation
        initializer: package.cache
        recurrence: {kind: invariant}
        service_group: decoder_cache
        management: runtime
        release_boundary: invocation
    serving:
      active: active
      done: done
      accepted_len: accepted_len
      state_service:
        groups:
          decoder_cache:
            kind: full_attention
            sequence_axis: 1
            layout: bsh
            logical_lengths: cache_lengths
            aliasing: permitted
            update:
              kind: indexed_scatter
              write_indices: cache_lengths
              capacity: package.capacity
              write_indices_ports:
                model: write_indices
              kv_length_ports:
                model: nonpad_kv_seqlen
            reuse: {prefix_reusable: true, evictable_prefix: true}
            capabilities: {rollback_positions: 8, snapshot: true, fork: true}
            ports:
              model:
                key_cache: {input: key_cache.0, output: updated_key_cache.0, role: key, layer: 0}
                value_cache: {input: value_cache.0, output: updated_value_cache.0, role: value, layer: 0}
    steps:
    - kind: loop
      setup: []
      steps:
      - kind: invoke
        component: model
        inputs:
          input_ids: token
          key_cache.0: key_cache
          value_cache.0: value_cache
          write_indices: cache_lengths
          nonpad_kv_seqlen: cache_lengths
        outputs:
          token: body.token
          updated_key_cache.0: body.key_cache
          updated_value_cache.0: body.value_cache
          next_write_indices: body.write_indices
          next_cache_lengths: body.cache_lengths
          next_active: body.active
          next_done: body.done
          next_accepted_len: body.accepted_len
          next_loop_active: body.loop_active
      - kind: emit
        value: body.token
        output: tokens
        mode: append
        when: active
      continue_when: loop_active
      max_iterations: request.max_iterations
      carried:
      - {cell: token, next: body.token}
      - {cell: key_cache, next: body.key_cache}
      - {cell: value_cache, next: body.value_cache}
      - {cell: write_indices, next: body.write_indices}
      - {cell: cache_lengths, next: body.cache_lengths}
      - {cell: active, next: body.active}
      - {cell: done, next: body.done}
      - {cell: accepted_len, next: body.accepted_len}
      - {cell: loop_active, next: body.loop_active}
      iteration:
        value: loop.iteration
        contract: {dtype: int64, rank: 1, shape: [1]}
package:
  tokenizer:
    algorithm: bpe
    vocab_size: 32
    byte_level: true
    artifacts:
    - location: tokenizer.json
"#;

fn workflow(dtype: &str) -> String {
    STATIC_CACHE_WORKFLOW.replace("DTYPE", dtype)
}

/// Rewrite a document through a generic YAML value.
///
/// `InferenceMetadata` is parse-only, so this is the round trip a package
/// actually takes: text in, text out, with key order and scalar formatting
/// chosen by the serializer rather than the author.
fn rewrite(document: &str) -> String {
    serde_yaml::to_string(&serde_yaml::from_str::<serde_yaml::Value>(document).expect("parses"))
        .expect("serializes")
}

fn cache_group(metadata: &InferenceMetadata) -> &onnx_genai_metadata::StateGroupContract {
    &metadata
        .pipeline
        .as_ref()
        .expect("pipeline")
        .workflow
        .serving
        .as_ref()
        .expect("serving")
        .state_service
        .groups["decoder_cache"]
}

/// The fixed-capacity declaration is accepted as written.
#[test]
fn indexed_scatter_group_validates() {
    let metadata = parse(&workflow("float32"));
    validate_metadata(&metadata).expect("a fixed-capacity cache is a legal declaration");
}

/// The parts of the declaration survive a round trip through YAML.
///
/// The update discipline is not derivable from anything else in the document, so
/// a serializer that dropped it would silently turn a static cache into an
/// append cache and the buffer would be read as if all of it were history.
#[test]
fn indexed_scatter_group_round_trips() {
    let document = workflow("float32");
    let rewritten = rewrite(&document);
    assert_eq!(
        semantic_identity_of_str(&document).expect("identity"),
        semantic_identity_of_str(&rewritten).expect("identity"),
        "rewriting the document changed what it means"
    );

    let metadata = parse(&rewritten);
    validate_metadata(&metadata).expect("the rewritten document is still legal");
    let Some(onnx_genai_metadata::StateUpdate::IndexedScatter {
        write_indices,
        capacity,
        write_indices_ports,
        ..
    }) = &cache_group(&metadata).update
    else {
        panic!("the update discipline did not survive the round trip");
    };
    assert_eq!(write_indices, "write_indices");
    assert_eq!(capacity, "package.capacity");
    assert_eq!(write_indices_ports["model"], "write_indices");
}

/// Without logical lengths the valid prefix of a fixed-capacity buffer is
/// unknowable: its shape is the capacity, and the slots past the end are
/// whatever was there before.
#[test]
fn indexed_scatter_requires_logical_lengths() {
    let document = workflow("float32").replace("            logical_lengths: cache_lengths\n", "");
    let errors = errors(&document);
    assert!(
        errors
            .iter()
            .any(|error| error.contains("logical_lengths") && error.contains("decoder_cache")),
        "{errors:?}"
    );
}

/// The write cursor has to be state, not a step output: it is restored with the
/// buffer it indexes, and a cursor that is not would point past what a restored
/// row actually holds.
#[test]
fn indexed_scatter_requires_a_declared_write_cursor() {
    let document = workflow("float32").replace(
        "              write_indices: write_indices\n",
        "              write_indices: body.write_indices\n",
    );
    let errors = errors(&document);
    assert!(
        errors
            .iter()
            .any(|error| error.contains("unknown write_indices state")),
        "{errors:?}"
    );
}

/// Destinations are an ordinary integer vector, indistinguishable from any other
/// integer control input, so the port carrying them cannot be inferred.
#[test]
fn indexed_scatter_requires_a_destination_port_for_every_bound_component() {
    let document = workflow("float32").replace(
        "              write_indices_ports:\n                model: write_indices\n",
        "              write_indices_ports: {}\n",
    );
    let errors = errors(&document);
    assert!(
        errors
            .iter()
            .any(|error| error.contains("declares no write_indices port")),
        "{errors:?}"
    );
}

/// A buffer whose capacity is fixed cannot also change shape; the destinations
/// are only meaningful against one constant extent.
#[test]
fn indexed_scatter_rejects_a_varying_buffer() {
    let document = workflow("float32").replace(
        "        recurrence: {kind: invariant}\n        service_group: decoder_cache",
        "        recurrence: {kind: bounded, axis: 1, max: package.capacity}\n        \
         service_group: decoder_cache",
    );
    let errors = errors(&document);
    assert!(
        errors
            .iter()
            .any(|error| error.contains("declares a varying shape")),
        "{errors:?}"
    );
}

/// An FP8 cache is a legal declaration.
///
/// Nothing in the contract depends on the element type: the capacity, the
/// cursor, and the valid prefix mean the same thing at one byte per element as
/// at four. Refusing FP8 here would be the schema answering a question about
/// kernels.
#[test]
fn fp8_state_group_validates() {
    for dtype in ["float8_e4m3fn", "float8_e5m2"] {
        let metadata = parse(&workflow(dtype));
        validate_metadata(&metadata)
            .unwrap_or_else(|errors| panic!("{dtype} state must validate: {errors:?}"));
    }
}

/// An FP8 cache keeps its exact dtype through a round trip.
///
/// A cache silently widened to float16 on the way out would still load and still
/// run; it would just use twice the memory the package asked for, and nothing
/// downstream would report it. So the test asserts the spelling, not merely that
/// the document survives.
#[test]
fn fp8_state_group_round_trips_without_widening() {
    for dtype in ["float8_e4m3fn", "float8_e5m2"] {
        let document = workflow(dtype);
        let rewritten = rewrite(&document);
        assert_eq!(
            semantic_identity_of_str(&document).expect("identity"),
            semantic_identity_of_str(&rewritten).expect("identity"),
            "{dtype} changed meaning across a round trip"
        );

        let metadata = parse(&rewritten);
        validate_metadata(&metadata)
            .unwrap_or_else(|errors| panic!("{dtype} must survive a round trip: {errors:?}"));
        let workflow = &metadata.pipeline.as_ref().expect("pipeline").workflow;
        assert_eq!(
            workflow.state["cache"].contract.dtype, dtype,
            "state widened"
        );
        assert_eq!(
            workflow.components["model"].ports.inputs["cache"].dtype, dtype,
            "port widened"
        );
        assert!(
            !rewritten.contains("float16") && !rewritten.contains("float32"),
            "{dtype} was widened somewhere in the document"
        );
    }
}

/// A workflow package declares its graph ABI once, in the workflow.
///
/// The canonical document carries no `model.io` at all: the ports are on the
/// component, the roles that name what they mean are beside them, and the cache
/// pairs and write discipline are on the state group. This test is the baseline
/// the rest of this section measures against.
#[test]
fn the_canonical_static_cache_package_needs_no_model_io() {
    let document = MOBIUS_STATIC_CACHE.replace("DTYPE", "float16");
    let metadata = parse(&document);
    validate_metadata(&metadata).expect("a workflow-only package is the canonical form");
    assert!(
        !document.contains("\nmodel:"),
        "the canonical package must not carry a second ABI declaration"
    );
}

/// A second serialized ABI beside the workflow is refused — at the loader,
/// before it can become a value.
///
/// Two declarations of the same graph are not redundancy but a fork: the moment
/// they disagree, which one a runtime obeys is decided by whichever code path
/// reached it first. There is now no schema field for the second one at all, so
/// the refusal happens where the document is read.
#[test]
fn model_io_beside_a_workflow_is_refused() {
    let document = format!(
        "{}model:\n  io:\n    token_input: input_ids\n",
        MOBIUS_STATIC_CACHE.replace("DTYPE", "float16")
    );
    let directory = tempfile::tempdir().expect("temp dir");
    let path = directory.path().join("inference_metadata.yaml");
    std::fs::write(&path, &document).expect("write");

    let error =
        onnx_genai_metadata::load_metadata(&path).expect_err("the retired block must not load");
    let message = error.to_string();
    assert!(
        message.contains("retired `model.io` block") && message.contains("migrate_model_io"),
        "the refusal must name what is wrong and how to fix it: {message}"
    );
}

/// The workflow alone yields the whole decode-step ABI.
///
/// This is the property that lets `model.io` go away: everything the optimized
/// single-decoder path used to read from a separate block is recoverable from
/// the component's ports, the roles beside them, and the state group.
#[test]
fn the_decode_abi_is_recognized_from_the_workflow_alone() {
    let metadata = parse(&MOBIUS_STATIC_CACHE.replace("DTYPE", "float16"));
    let io = metadata
        .decoder_io()
        .expect("the workflow declares one decoder");

    assert_eq!(io.token_input.as_deref(), Some("input_ids"));
    assert_eq!(
        io.kv_ownership,
        Some(onnx_genai_metadata::KvOwnership::Owned)
    );
    // A fixed-capacity cache is described by the static-cache ABI below and by
    // nothing else: reporting the same buffers as growing past/present pairs
    // would be two answers for one cache, and the paged KV bridge reads their
    // absence as "this cache does not grow".
    assert_eq!(io.kv_inputs, None);
    assert_eq!(io.kv_outputs, None);

    let static_cache = io
        .static_cache
        .as_ref()
        .expect("an indexed_scatter group is a static cache");
    assert_eq!(static_cache.write_indices_input, "write_indices");
    assert_eq!(static_cache.kv_sequence_length_input, "nonpad_kv_seqlen");
    assert_eq!(static_cache.key_cache_inputs, ["key_cache.0"]);
    assert_eq!(static_cache.value_cache_inputs, ["value_cache.0"]);
    assert_eq!(static_cache.key_cache_outputs, ["updated_key_cache.0"]);
    assert_eq!(static_cache.value_cache_outputs, ["updated_value_cache.0"]);
}

/// The recognized ABI is exactly what the deleted block used to say.
///
/// A migration that produced a *different* ABI would be a silent behavior
/// change dressed up as a cleanup, so the equivalence is asserted against the
/// literal block this package used to carry.
#[test]
fn the_recognized_abi_equals_the_block_it_replaced() {
    let metadata = parse(&MOBIUS_STATIC_CACHE.replace("DTYPE", "float16"));
    let recognized = metadata.decoder_io().expect("recognized");
    let retired: onnx_genai_metadata::StaticCacheIoSpec = serde_yaml::from_str(
        "write_indices_input: write_indices\n\
         kv_sequence_length_input: nonpad_kv_seqlen\n\
         key_cache_inputs: [key_cache.0]\n\
         value_cache_inputs: [value_cache.0]\n\
         key_cache_outputs: [updated_key_cache.0]\n\
         value_cache_outputs: [updated_value_cache.0]\n",
    )
    .expect("the retired block parses");
    assert_eq!(recognized.static_cache.as_ref(), Some(&retired));
}

/// Split cache halves are ordered by declared layer, never by label order.
///
/// A producer that labels layers `layer.2` and `layer.10` gets lexicographic
/// order from the map, which would positionally pair layer 10's keys with layer
/// 2's values and corrupt attention in a way no shape check can see.
#[test]
fn per_layer_cache_order_follows_the_declared_layer_index() {
    let document = MOBIUS_STATIC_CACHE.replace("DTYPE", "float16").replace(
        "key_cache: {input: key_cache.0, output: updated_key_cache.0, role: key, layer: 0}",
        "zz_late_label: {input: key_cache.0, output: updated_key_cache.0, role: key, layer: 0}",
    );
    let metadata = parse(&document);
    let io = metadata.decoder_io().expect("recognized");
    assert_eq!(
        io.static_cache
            .as_ref()
            .expect("static cache")
            .key_cache_inputs,
        ["key_cache.0"],
        "the declared layer index, not the label, decides the order"
    );
}

/// One cell may serve as both the write cursor and the valid length.
///
/// A row's next write lands exactly where its valid prefix ends, so a second
/// carried cell would only ever hold a copy. Refusing this would force an
/// exporter to invent a value nothing reads independently.
#[test]
fn the_write_cursor_and_the_valid_length_may_be_one_cell() {
    let metadata = parse(&MOBIUS_STATIC_CACHE.replace("DTYPE", "float16"));
    let Some(onnx_genai_metadata::StateUpdate::IndexedScatter { write_indices, .. }) =
        &cache_group(&metadata).update
    else {
        panic!("expected an indexed_scatter group");
    };
    assert_eq!(write_indices, "cache_lengths");
    assert_eq!(
        cache_group(&metadata).logical_lengths.as_deref(),
        Some("cache_lengths"),
        "the cursor and the length are the same quantity in this package"
    );
}

/// A cache ABI is refused when it is declared halfway.
///
/// A group whose ports carry key/value roles is advertising the per-layer
/// buffers a direct driver binds positionally. Dropping the length port leaves
/// that advertisement unsatisfiable: destinations without a valid prefix cannot
/// tell a graph how much of a capacity-sized buffer to attend over. The failure
/// mode this prevents is the quiet one — `decoder_io()` returning no static
/// cache and the package looking merely unfeatured rather than faulty.
#[test]
fn a_cache_abi_missing_its_length_port_is_refused() {
    let document = MOBIUS_STATIC_CACHE.replace("DTYPE", "float16").replace(
        "              kv_length_ports:\n                model: nonpad_kv_seqlen\n",
        "",
    );
    let errors = errors(&document);
    assert!(
        errors
            .iter()
            .any(|error| error.contains("declares no kv_length port")),
        "{errors:?}"
    );
}

/// Per-layer buffers must say which layer they are.
///
/// The binding label is producer-chosen and its lexicographic order is not the
/// layer order, so a second key port with no layer index would be placed by
/// position. Two transposed caches have identical shapes and dtypes, so nothing
/// downstream detects the swap — the model just produces subtly wrong tokens.
#[test]
fn several_ports_of_one_role_must_declare_their_layers() {
    let document = MOBIUS_STATIC_CACHE.replace("DTYPE", "float16").replace(
        "                key_cache: {input: key_cache.0, output: updated_key_cache.0, role: key, layer: 0}",
        "                key_cache: {input: key_cache.0, output: updated_key_cache.0, role: key}\n                \
         key_cache_1: {input: key_cache.1, output: updated_key_cache.1, role: key}",
    );
    let errors = errors(&document);
    assert!(
        errors
            .iter()
            .any(|error| error.contains("declare no layer")),
        "{errors:?}"
    );
}

/// Two buffers may not claim one layer.
///
/// Positional binding would keep whichever the sort happened to place last, so
/// one layer's cache would silently shadow the other's.
#[test]
fn two_ports_may_not_claim_the_same_layer() {
    let document = MOBIUS_STATIC_CACHE.replace("DTYPE", "float16").replace(
        "                key_cache: {input: key_cache.0, output: updated_key_cache.0, role: key, layer: 0}",
        "                key_cache: {input: key_cache.0, output: updated_key_cache.0, role: key, layer: 0}\n                \
         key_cache_1: {input: key_cache.1, output: updated_key_cache.1, role: key, layer: 0}",
    );
    let errors = errors(&document);
    assert!(
        errors.iter().any(|error| error.contains("already claims")),
        "{errors:?}"
    );
}

fn add_cache_alias(metadata: &mut InferenceMetadata, source: &str, label: &str, layer: usize) {
    let bindings = metadata
        .pipeline
        .as_mut()
        .unwrap()
        .workflow
        .serving
        .as_mut()
        .unwrap()
        .state_service
        .groups
        .get_mut("decoder_cache")
        .unwrap()
        .ports
        .get_mut("model")
        .unwrap();
    let mut alias = bindings.get(source).unwrap().clone();
    alias.layer = Some(layer);
    bindings.insert(label.to_string(), alias);
}

#[test]
fn split_attention_cache_rejects_different_key_and_value_counts() {
    let mut metadata = parse(&MOBIUS_STATIC_CACHE.replace("DTYPE", "float16"));
    add_cache_alias(&mut metadata, "key_cache", "key_cache_1", 1);

    let errors = validate_metadata(&metadata).expect_err("unpaired key layer must be rejected");
    assert!(
        errors.iter().any(|error| {
            error.contains("key layers are {0, 1}")
                && error.contains("value layers are {0}")
                && error.contains("missing value layers are [1]")
        }),
        "{errors:?}"
    );
}

#[test]
fn split_attention_cache_rejects_mismatched_layer_sets() {
    let mut metadata = parse(&MOBIUS_STATIC_CACHE.replace("DTYPE", "float16"));
    add_cache_alias(&mut metadata, "key_cache", "key_cache_1", 1);
    add_cache_alias(&mut metadata, "value_cache", "value_cache_2", 2);

    let errors = validate_metadata(&metadata).expect_err("mismatched layers must be rejected");
    assert!(
        errors.iter().any(|error| {
            error.contains("key layers are {0, 1}")
                && error.contains("value layers are {0, 2}")
                && error.contains("missing value layers are [1]")
                && error.contains("missing key layers are [2]")
        }),
        "{errors:?}"
    );
}

#[test]
fn matching_layers_allow_heterogeneous_key_and_value_geometry() {
    let mut metadata = parse(&MOBIUS_STATIC_CACHE.replace("DTYPE", "float16"));
    {
        let workflow = &mut metadata.pipeline.as_mut().unwrap().workflow;
        let value_shape = vec![
            TensorDimension::Symbol("batch".to_string()),
            TensorDimension::Symbol("cache_capacity".to_string()),
            TensorDimension::Fixed(8),
        ];
        let mut value_initializer = workflow.inputs["package.cache"].clone();
        value_initializer.contract.shape = Some(value_shape.clone());
        workflow
            .inputs
            .insert("package.value_cache".to_string(), value_initializer);
        workflow
            .state
            .get_mut("value_cache")
            .unwrap()
            .contract
            .shape = Some(value_shape.clone());
        workflow.state.get_mut("value_cache").unwrap().initializer =
            "package.value_cache".to_string();
        let ports = &mut workflow.components.get_mut("model").unwrap().ports;
        ports.inputs.get_mut("value_cache.0").unwrap().shape = Some(value_shape.clone());
        ports
            .outputs
            .get_mut("updated_value_cache.0")
            .unwrap()
            .shape = Some(value_shape);
    }

    validate_metadata(&metadata)
        .expect("matching key/value layer identity must not require equal cache geometry");
}

/// The canonical static-cache package validates as a whole package, on disk.
///
/// Every other test here validates a YAML string. That leaves the property a
/// producer actually depends on untested: that the package *directory* — with
/// its ONNX artifact resolved and inspected — passes the same
/// `load_metadata_package` entry point the `validate_metadata` binary calls.
/// The fixture is the worked example for adopting the canonical form, so it has
/// to survive the path a producer will take, not just the one the unit tests do.
///
/// Its shape is deliberately the minimum: no top-level `model:` block, one
/// ONNX-backed component whose only declaration is a single `ports.roles` entry,
/// and a state service that carries the whole fixed-capacity ABI.
#[test]
fn the_canonical_static_cache_package_validates_on_disk() {
    let package = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-llm-scatter-workflow");
    let metadata = onnx_genai_metadata::load_metadata_package(&package)
        .expect("the canonical package validates as a directory, artifact included");

    // No second ABI: the package carries no `model:` block at all.
    assert!(metadata.model.is_none());

    // The static-cache ABI is still fully resolved, derived from the workflow.
    let io = metadata
        .decoder_io()
        .expect("the workflow alone resolves the decoder ABI");
    let cache = io
        .static_cache
        .as_ref()
        .expect("the state service supplies the fixed-capacity ABI");
    assert_eq!(cache.write_indices_input, "write_indices");
    assert_eq!(cache.kv_sequence_length_input, "nonpad_kv_seqlen");
    assert_eq!(cache.key_cache_inputs.len(), cache.value_cache_inputs.len());
    assert_eq!(cache.key_cache_inputs.len(), cache.key_cache_outputs.len());
}

/// Loading a package enforces the document-level invariants, not just the
/// pipeline-scoped ones.
///
/// `load_metadata_package` used to run only `validate_pipeline_spec`, which by
/// construction cannot see a rule about `model` and `pipeline` together. The
/// ban on carrying both a `model.io` block and a workflow therefore held only
/// for callers who reached for `validate_metadata` directly — which excluded
/// every producer loading a package from disk, and excluded the
/// `validate_metadata` binary itself. A package with two conflicting ABIs
/// reported `valid`, and at runtime `decoder_io()` silently resolved the
/// workflow and ignored the `model.io` the producer believed it had published.
/// That is exactly the divergence the rule exists to prevent, so the rule has
/// to be reachable from the path producers actually run.
#[test]
fn loading_a_package_rejects_a_second_serialized_abi() {
    let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-llm-scatter-workflow");
    let staged = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("two-abi-package");
    std::fs::create_dir_all(&staged).expect("staging directory");
    std::fs::copy(
        source.join("model.onnx.textproto"),
        staged.join("model.onnx.textproto"),
    )
    .expect("artifact copy");

    // The canonical package, plus the one thing it is not allowed to also carry.
    let canonical = std::fs::read_to_string(source.join("inference_metadata.yaml"))
        .expect("canonical metadata");
    std::fs::write(
        staged.join("inference_metadata.yaml"),
        format!("model:\n  io:\n    token_input: input_ids\n{canonical}"),
    )
    .expect("staged metadata");

    let error = onnx_genai_metadata::load_metadata_package(&staged)
        .expect_err("a package carrying both a model.io block and a workflow is rejected");
    let rendered = error.to_string();
    assert!(
        rendered.contains("retired `model.io` block") && rendered.contains("migrate_model_io"),
        "the rejection must name the retired block and the conversion, got: {rendered}"
    );

    // The same package without the second ABI is the canonical form and loads.
    onnx_genai_metadata::load_metadata_package(&source)
        .expect("the workflow-only package is the canonical form");
}

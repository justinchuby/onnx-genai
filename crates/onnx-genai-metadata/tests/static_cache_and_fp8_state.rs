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

use onnx_genai_metadata::{InferenceMetadata, semantic_identity_of_str, validate_metadata};

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
      ir_version: '1.0'
      onnx_opsets:
        ai.onnx: 24
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
      sha256: '0000000000000000000000000000000000000000000000000000000000000000'
    special_tokens:
      bos: {id: 1, content: <s>}
      eos: {id: 2, content: </s>}
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

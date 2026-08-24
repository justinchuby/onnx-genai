//! Invariants the metadata redesign must keep true.
//!
//! Each test here pins one contract that is easy to break silently: a fact that
//! disappears, a check that reads the wrong field, or a runtime-owned decision
//! leaking back into the portable document. The fixtures are deliberately small
//! so the fact under test is the only thing that varies.

use onnx_genai_metadata::{
    InferenceMetadata, WorkflowOutputRole, cache_dependencies, semantic_identity_of_str,
    validate_metadata,
};

fn parse(document: &str) -> InferenceMetadata {
    serde_yaml::from_str(document).expect("metadata parses")
}

fn errors(document: &str) -> Vec<String> {
    validate_metadata(&parse(document)).expect_err("metadata must be rejected")
}

#[test]
fn image_outputs_require_an_explicit_value_range() {
    let document = include_str!(
        "../../../examples/inference_metadata/catalogue/07-stable-diffusion-text-to-image.yaml"
    );
    let mut metadata = parse(document);
    metadata
        .pipeline
        .as_mut()
        .expect("catalogue entry has a pipeline")
        .workflow
        .outputs
        .get_mut("image")
        .expect("catalogue entry has an image output")
        .value_range = None;
    let errors = validate_metadata(&metadata).expect_err("metadata must be rejected");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("image output 'image' must declare value_range")),
        "{errors:#?}"
    );
}

#[test]
fn image_value_range_is_rejected_on_non_image_outputs() {
    let document = include_str!(
        "../../../examples/inference_metadata/catalogue/07-stable-diffusion-text-to-image.yaml"
    );
    let mut metadata = parse(document);
    metadata
        .pipeline
        .as_mut()
        .expect("catalogue entry has a pipeline")
        .workflow
        .outputs
        .get_mut("image")
        .expect("catalogue entry has an image output")
        .role = WorkflowOutputRole::Tensor;
    let errors = validate_metadata(&metadata).expect_err("metadata must be rejected");
    assert!(
        errors.iter().any(
            |error| error.contains("non-image output 'image' cannot declare image value_range")
        ),
        "{errors:#?}"
    );
}

#[test]
fn workflow_manifest_rejects_facts_owned_by_schema_and_onnx_artifacts() {
    for duplicated in ["ir_version: '1.0'", "onnx_opsets: {ai.onnx: 24}"] {
        let document = format!(
            "schema_version: v1\npipeline:\n  workflow:\n    manifest:\n      \
             {duplicated}\n    components: {{}}\n    steps: []\n"
        );
        let error = serde_yaml::from_str::<InferenceMetadata>(&document)
            .expect_err("duplicated manifest fact must be rejected")
            .to_string();
        assert!(error.contains("unknown field"), "{error}");
    }
}

/// A serving decoder with a LoRA service, an externally-suppliable encoder
/// result, a native grammar component, and a generation-affecting profile.
/// Every one of those is a cache correctness dependency.
const MULTIMODAL_ADAPTER_WORKFLOW: &str = r#"
schema_version: v1
adapters:
  target_manifest:
    targets:
      - id: projection
        component: decoder
        initializer: projection
        layer_index: 0
        node_name: projection
        output_name: projection.output
        activation_dtype: float32
        input_features: 2
        output_features: 2
        rank: 1
        alpha: 1.0
  selection:
    segments: request.adapter_segments
    adapter_counts: request.adapter_counts
    scales: request.adapter_scales
    max_adapters: 2
  application_capability: onnx-genai.adapters@1
  portable_fallback: true
  artifacts:
    red:
      index: 0
      identity: red
      version: "1"
      rank: 1
      alpha: 1.0
      dtype: float32
      weights:
        - location: adapters/red/adapter.json
          loader_capability: onnx-genai.adapters.json@1
          scale_encoding: alpha_over_rank
          format: json
      bindings:
        - target: projection
          weight_key: projection
profiles:
  chat:
    kind: generation
    version: "1"
    generation_affecting: true
pipeline:
  workflow:
    manifest:
      adapter_abis: { onnx-genai.parameter-overlay: "1" }
      capabilities: [workflow_ssa, linear_effects, typed_emit, parameter_adapters, heterogeneous_adapter_batching]
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
      vision.embeddings:
        contract: { dtype: float32, rank: 3, shape: [batch, tiles, hidden], batch_layout: { kind: request_aligned, axis: 0 } }
        role: { kind: opaque }
        source: { kind: application, name: vision.embeddings }
        externally_suppliable: true
      image_features:
        contract:
          dtype: float32
          rank: 2
          shape: [items, hidden]
          batch_layout: { kind: token_packed, axis: 0, levels: [{ offsets: image_offsets, owner: image_owner }] }
        role: { kind: opaque }
        source: { kind: application, name: image_features }
        externally_suppliable: true
      image_offsets:
        contract: { dtype: int64, rank: 1, shape: [rows_plus_one], batch_layout: { kind: shared } }
        role: { kind: opaque }
        source: { kind: application, name: image_offsets }
      image_owner:
        contract: { dtype: int64, rank: 1, shape: [items], batch_layout: { kind: shared } }
        role: { kind: opaque }
        source: { kind: application, name: image_owner }
      prompt:
        contract: { dtype: int64, rank: 2, shape: [batch, sequence], batch_layout: { kind: request_aligned, axis: 0 } }
        role: { kind: opaque }
        source: { kind: application, name: prompt }
    outputs:
      tokens:
        contract: { dtype: int64, rank: 2, shape: [batch, generated], batch_layout: { kind: request_aligned, axis: 0 } }
        role: tokens
        stage: pre_adapter
    effects:
      grammar:
        retry: idempotent
        speculation_safety: { kind: rewindable, max_depth: 8 }
    components:
      splice:
        implementation: { kind: onnx, artifact: splice.onnx }
        ports:
          inputs:
            prompt: { dtype: int64, rank: 2, shape: [batch, sequence], batch_layout: { kind: request_aligned, axis: 0 } }
            embeddings: { dtype: float32, rank: 3, shape: [batch, tiles, hidden], batch_layout: { kind: request_aligned, axis: 0 } }
          outputs:
            spliced: { dtype: float32, rank: 3, shape: [batch, sequence, hidden], batch_layout: { kind: request_aligned, axis: 0 } }
      decoder:
        implementation: { kind: onnx, artifact: decoder.onnx }
        ports:
          inputs:
            hidden: { dtype: float32, rank: 3, shape: [batch, sequence, hidden], batch_layout: { kind: request_aligned, axis: 0 } }
          outputs:
            token: { dtype: int64, rank: 2, shape: [batch, generated], batch_layout: { kind: request_aligned, axis: 0 } }
      grammar:
        implementation: { kind: binding }
        row_scope: { axis: 0, stateful: true }
        effects: [grammar]
        cache_affects_state: [grammar.parser_table]
        ports:
          inputs:
            token: { dtype: int64, rank: 2, shape: [batch, generated], batch_layout: { kind: request_aligned, axis: 0 } }
          outputs:
            guided: { dtype: int64, rank: 2, shape: [batch, generated], batch_layout: { kind: request_aligned, axis: 0 } }
    steps:
      - kind: invoke
        component: splice
        inputs: { prompt: prompt, embeddings: vision.embeddings }
        outputs: { spliced: hidden }
      - kind: invoke
        component: decoder
        inputs: { hidden: hidden }
        outputs: { token: raw }
      - kind: invoke
        component: grammar
        inputs: { token: raw }
        outputs: { guided: guided }
      - kind: emit
        value: guided
        output: tokens
        mode: replace
"#;

#[test]
fn cache_dependencies_cannot_omit_lora_multimodal_or_profile_facts() {
    let metadata = parse(MULTIMODAL_ADAPTER_WORKFLOW);
    validate_metadata(&metadata).expect("fixture is valid");
    let dependencies = cache_dependencies(&metadata);

    // LoRA: activating a different adapter changes the result, so the artifact
    // identity and version are dependencies.
    assert!(
        dependencies.adapters.contains("red@red:1"),
        "{:?}",
        dependencies.adapters
    );

    // Multimodal: an externally-suppliable encoder result is spliced into the
    // decode. Two requests with the same tokens and different images must not
    // share a cache entry.
    assert!(
        dependencies.inputs.contains("vision.embeddings"),
        "{:?}",
        dependencies.inputs
    );

    // Transitive component dataflow: the splice feeds the decoder, so its
    // implementation identity is a dependency even though it emits nothing.
    assert!(
        dependencies.components.contains("splice=onnx:splice.onnx"),
        "{:?}",
        dependencies.components
    );
    assert!(
        dependencies
            .components
            .contains("decoder=onnx:decoder.onnx"),
        "{:?}",
        dependencies.components
    );

    // A native component with no dataflow edge to its table still declares the
    // non-dataflow state it reads.
    assert!(
        dependencies.external_state.contains("grammar.parser_table"),
        "{:?}",
        dependencies.external_state
    );

    // A generation-affecting profile changes generated output.
    assert!(
        dependencies.profiles.contains("chat@1"),
        "{:?}",
        dependencies.profiles
    );
}

#[test]
fn cache_dependencies_shrink_when_a_contributing_fact_is_removed() {
    let metadata = parse(MULTIMODAL_ADAPTER_WORKFLOW);
    let full = cache_dependencies(&metadata);

    // Dropping the adapter service drops exactly the adapter facts. This is the
    // regression guard: a future refactor that stops walking the adapter
    // manifest would make these two sets equal.
    let mut without_adapters = metadata.clone();
    without_adapters.adapters = None;
    let reduced = cache_dependencies(&without_adapters);
    assert!(reduced.adapters.is_empty());
    assert_ne!(full.adapters, reduced.adapters);
    assert_eq!(full.inputs, reduced.inputs);

    // Marking the encoder result as not externally suppliable drops exactly the
    // multimodal input fact.
    let mut without_vision = metadata.clone();
    without_vision
        .pipeline
        .as_mut()
        .expect("pipeline")
        .workflow
        .inputs
        .get_mut("vision.embeddings")
        .expect("vision input")
        .externally_suppliable = false;
    let reduced = cache_dependencies(&without_vision);
    assert!(!reduced.inputs.contains("vision.embeddings"));
    assert_eq!(full.components, reduced.components);

    // A profile that does not affect generation is not a dependency.
    let mut inert_profile = metadata;
    inert_profile
        .profiles
        .get_mut("chat")
        .expect("profile")
        .generation_affecting = false;
    assert!(cache_dependencies(&inert_profile).profiles.is_empty());
}

/// A minimal serving workflow with one state group, parameterized so each test
/// can vary exactly one fact.
fn serving_workflow(
    aliasing: &str,
    capabilities: &str,
    effects: &str,
    speculative: &str,
) -> String {
    let linear_effects = if effects.is_empty() {
        ""
    } else {
        ", linear_effects"
    };
    format!(
        r#"
pipeline:
  workflow:
    manifest:
      capabilities: [workflow_ssa, serving_service_contract{linear_effects}]
    inputs:
      active:
        contract: {{ dtype: bool, rank: 1, shape: [batch], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: active }}
      done:
        contract: {{ dtype: bool, rank: 1, shape: [batch], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: done }}
      accepted_len:
        contract: {{ dtype: int64, rank: 1, shape: [batch], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: accepted_len }}
      empty_cache:
        contract: {{ dtype: float16, rank: 4, shape: [batch, heads, sequence, head_dim], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: empty_cache }}
    components:
      proposer:
        implementation: {{ kind: onnx, artifact: proposer.onnx }}
        ports: {{}}
{effects}
      verifier:
        implementation: {{ kind: onnx, artifact: verifier.onnx }}
        ports: {{}}
    state:
      cache:
        contract: {{ dtype: float16, rank: 4, shape: [batch, heads, sequence, head_dim], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        class: semantic
        scope: invocation
        initializer: empty_cache
        recurrence: {{ kind: invariant }}
        service_group: decoder_cache
        release_boundary: invocation
        management: runtime
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
            aliasing: {aliasing}
            reuse: {{ prefix_reusable: true, evictable_prefix: true }}
            capabilities: {capabilities}
            ports:
              verifier:
                cache: {{ input: past_key_values, output: present_key_values }}
    steps:
      - kind: invoke
        component: proposer
      - kind: invoke
        component: verifier
{speculative}
"#
    )
}

const SOUND_CAPABILITIES: &str = "{ rollback_positions: 8, snapshot: true, fork: true }";

const REWINDABLE_GRAMMAR: &str = r#"    effects:
      grammar:
        retry: idempotent
        speculation_safety: { kind: rewindable, max_depth: 8 }
    components:"#;

fn speculative_block(width: usize) -> String {
    format!(
        r#"
speculative:
  proposer: proposer
  target: verifier
  vocabulary: {{ kind: identical }}
  max_proposal_width: {width}
  shared_state: [cache]
  shared_weights: []
  distribution_preserving: true
  rollback_state: [cache]
"#
    )
}

#[test]
fn speculative_rollback_reads_speculation_safety_not_the_retry_class() {
    // An effect can be perfectly safe to retry and still impossible to rewind.
    // The rollback check must read speculation_safety, never the retry class.
    let workflow = serving_workflow(
        "permitted",
        SOUND_CAPABILITIES,
        r#"        effects: [grammar]"#,
        &speculative_block(4),
    );
    let unrewindable = workflow.replace(
        "    components:",
        r#"    effects:
      grammar:
        retry: idempotent
        speculation_safety: { kind: none }
    components:"#,
    );
    let reported = errors(&unrewindable);
    assert!(
        reported
            .iter()
            .any(|error| error.contains("speculation_safety none")),
        "{reported:?}"
    );

    // The same idempotent retry class with a rewindable bound is accepted.
    let rewindable = workflow.replace("    components:", REWINDABLE_GRAMMAR);
    validate_metadata(&parse(&rewindable)).expect("rewindable effect is speculative-safe");

    // A retry class that sounds unsafe does not by itself block speculation.
    let non_retryable = rewindable.replace("retry: idempotent", "retry: non_retryable");
    validate_metadata(&parse(&non_retryable))
        .expect("retry class must not be read as a speculation bound");
}

#[test]
fn speculative_rollback_bounds_must_cover_the_maximum_proposal_width() {
    let too_narrow = serving_workflow(
        "permitted",
        "{ rollback_positions: 2, snapshot: true, fork: true }",
        r#"        effects: [grammar]"#,
        &speculative_block(4),
    )
    .replace("    components:", REWINDABLE_GRAMMAR);
    let reported = errors(&too_narrow);
    assert!(
        reported
            .iter()
            .any(|error| error.contains("rolls back 2 positions")),
        "{reported:?}"
    );

    let effect_too_narrow = serving_workflow(
        "permitted",
        SOUND_CAPABILITIES,
        r#"        effects: [grammar]"#,
        &speculative_block(4),
    )
    .replace(
        "    components:",
        r#"    effects:
      grammar:
        retry: idempotent
        speculation_safety: { kind: rewindable, max_depth: 2 }
    components:"#,
    );
    let reported = errors(&effect_too_narrow);
    assert!(
        reported.iter().any(|error| error.contains("rewinds 2")),
        "{reported:?}"
    );
}

#[test]
fn chained_proposer_requires_typed_ports_and_rollbackable_recurrence() {
    let metadata = serving_workflow(
        "permitted",
        SOUND_CAPABILITIES,
        "",
        r#"
speculative:
  proposer: proposer
  target: verifier
  proposal_execution:
    kind: chained
    token_embedding_input: inputs_embeds
    logits_output: draft_logits
    recurrent:
    - { state: cache, input: past_state, output: next_state }
  vocabulary: { kind: mapped, artifact: draft_to_target.npy }
  max_proposal_width: 4
  distribution_preserving: true
  rollback_state: [cache]
"#,
    )
    .replacen(
        "        ports: {}",
        r#"        ports:
          inputs:
            inputs_embeds: { dtype: float16, rank: 4, shape: [batch, heads, sequence, head_dim] }
            past_state: { dtype: float16, rank: 4, shape: [batch, heads, sequence, head_dim] }
          outputs:
            draft_logits: { dtype: float16, rank: 4, shape: [batch, heads, sequence, head_dim] }
            next_state: { dtype: float16, rank: 4, shape: [batch, heads, sequence, head_dim] }"#,
        1,
    )
    .replacen(
        "      - kind: invoke\n        component: proposer",
        "      - kind: invoke\n        component: proposer\n        inputs: { inputs_embeds: empty_cache, past_state: empty_cache }\n        outputs: { draft_logits: draft.logits, next_state: draft.next_state }",
        1,
    )
    // The proposer's recurrence advances `cache`, so the decoder_cache group
    // must expose a read_write proposer alias for it: the verifier alias alone
    // does not carry the proposer's loop, and a chained recurrence must resolve
    // through serving.state_service.groups.*.ports.proposer.
    .replace(
        "              verifier:\n                cache: { input: past_key_values, output: present_key_values }",
        "              verifier:\n                cache: { input: past_key_values, output: present_key_values }\n              proposer:\n                cache: { input: past_state, output: next_state }",
    );
    validate_metadata(&parse(&metadata)).expect("typed chained proposer must validate");

    let missing_rollback = metadata.replace("  rollback_state: [cache]", "  rollback_state: []");
    let reported = errors(&missing_rollback);
    assert!(
        reported
            .iter()
            .any(|error| error.contains("must be listed in rollback_state")),
        "{reported:?}"
    );

    let bad_port = metadata.replace("logits_output: draft_logits", "logits_output: missing");
    let reported = errors(&bad_port);
    assert!(
        reported
            .iter()
            .any(|error| error.contains("is not an output port")),
        "{reported:?}"
    );
}

#[test]
fn past_present_alias_legality_survives_the_removal_of_storage_policy() {
    // `shared_buffer` was a storage decision. What it also carried — whether the
    // graph tolerates a past/present alias — is a real graph ABI fact and must
    // still be expressible in all three states.
    for aliasing in ["permitted", "required", "forbidden"] {
        let document = serving_workflow(aliasing, SOUND_CAPABILITIES, "", "");
        let metadata = parse(&document);
        validate_metadata(&metadata).expect("aliasing legality is a declarable fact");
        let group = &metadata
            .pipeline
            .as_ref()
            .expect("pipeline")
            .workflow
            .serving
            .as_ref()
            .expect("serving")
            .state_service
            .groups["decoder_cache"];
        assert_eq!(
            serde_yaml::to_value(group.aliasing).expect("aliasing serializes"),
            serde_yaml::Value::String(aliasing.into())
        );
    }

    // The default is the safe one: a package that says nothing must not be read
    // as permitting an alias.
    let defaulted = serving_workflow("forbidden", SOUND_CAPABILITIES, "", "")
        .replace("            aliasing: forbidden\n", "");
    let metadata = parse(&defaulted);
    assert_eq!(
        metadata
            .pipeline
            .as_ref()
            .expect("pipeline")
            .workflow
            .serving
            .as_ref()
            .expect("serving")
            .state_service
            .groups["decoder_cache"]
            .aliasing,
        onnx_genai_metadata::StateAliasing::Forbidden
    );
}

#[test]
fn runtime_owned_storage_policy_cannot_reenter_the_document() {
    // These were allocator and placement decisions. They belong to the runtime,
    // and a package that still ships them must fail closed rather than have them
    // quietly ignored.
    for retired in [
        "            storage: shared_buffer\n",
        "            compaction: slot_permutation\n",
        "            paging: paged\n",
        "            allocation: runtime\n",
    ] {
        let document = serving_workflow("permitted", SOUND_CAPABILITIES, "", "").replace(
            "            layout: bnsh\n",
            &format!("            layout: bnsh\n{retired}"),
        );
        assert!(
            serde_yaml::from_str::<InferenceMetadata>(&document).is_err(),
            "runtime storage policy must be rejected: {retired}"
        );
    }
}

#[test]
fn graph_visible_state_representation_is_separate_from_runtime_storage() {
    // What the graph sees is metadata: the port dtype the state cell declares,
    // the sequence axis, and the layout. How the runtime stores or quantizes
    // the same cache in its own memory is not.
    let metadata = parse(&serving_workflow("permitted", SOUND_CAPABILITIES, "", ""));
    let workflow = &metadata.pipeline.as_ref().expect("pipeline").workflow;
    assert_eq!(workflow.state["cache"].contract.dtype, "float16");
    let group = &workflow
        .serving
        .as_ref()
        .expect("serving")
        .state_service
        .groups["decoder_cache"];
    assert_eq!(group.sequence_axis, Some(2));
    assert_eq!(group.layout, "bnsh");

    // Runtime-private cache representation cannot be declared anywhere. The
    // `model.io` spellings are absent from this list because the whole block is
    // retired: a document carrying it is refused at the loader with an error
    // naming the conversion, which is a stronger rejection than an unknown
    // field inside a block that no longer exists.
    for retired in [
        "kv_cache: { native_dtype: float16 }\n",
        "model:\n  runtime_configurable:\n    kv_cache: { dtype: [float16] }\n",
    ] {
        assert!(
            serde_yaml::from_str::<InferenceMetadata>(retired).is_err(),
            "runtime-private cache representation must be rejected: {retired}"
        );
    }

    // Model-weight quantization intent is unaffected: it is a property of the
    // shipped weights, not of a runtime's private cache.
    let weights: InferenceMetadata = serde_yaml::from_str("quantization:\n  default: int4\n")
        .expect("weight quantization intent survives");
    assert_eq!(
        weights
            .quantization
            .expect("quantization")
            .default
            .as_deref(),
        Some("int4")
    );
}

#[test]
fn equivalence_class_gates_automatic_implementation_substitution() {
    use onnx_genai_metadata::EquivalenceClass;

    // A runtime may substitute an implementation freely, but only a
    // distribution-preserving contract may be swapped in automatically for a
    // speculative optimization; anything weaker needs the caller to opt in.
    assert!(EquivalenceClass::Bitwise.permits_automatic_speculation());
    assert!(EquivalenceClass::DistributionPreserving.permits_automatic_speculation());
    assert!(!EquivalenceClass::Semantic.permits_automatic_speculation());

    // The default is the conservative one: a contract that says nothing must not
    // be read as safe to swap silently.
    assert_eq!(EquivalenceClass::default(), EquivalenceClass::Semantic);

    // A shipped package declares the class on the contract, so a runtime knows
    // before substituting anything whether the swap is silently allowed.
    let package = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/onnx_genai_workflows/speculative/inference_metadata.yaml");
    let metadata: InferenceMetadata =
        serde_yaml::from_str(&std::fs::read_to_string(package).expect("fixture"))
            .expect("fixture parses");
    let workflow = &metadata.pipeline.as_ref().expect("pipeline").workflow;
    let classes = workflow
        .components
        .values()
        .filter_map(|component| component.contract.as_ref())
        .map(|contract| contract.equivalence)
        .collect::<std::collections::BTreeSet<_>>();
    assert!(classes.contains(&EquivalenceClass::Bitwise), "{classes:?}");
}

#[test]
fn generation_overrides_are_structural_and_fail_loud() {
    let base = r#"
generation:
  defaults:
    temperature: 0.7
  overrides:
    temperature: { input: request.temperature }
pipeline:
  workflow:
    manifest:
      capabilities: [workflow_ssa]
    inputs:
      request.temperature:
        contract: { dtype: float32, rank: 1, shape: [batch], batch_layout: { kind: request_aligned, axis: 0 } }
        role: { kind: runtime, version: "1.0", role: sampling_temperature }
        source: { kind: request }
    components:
      decoder:
        implementation: { kind: onnx, artifact: decoder.onnx }
        ports: {}
    steps:
      - kind: invoke
        component: decoder
"#;
    validate_metadata(&parse(base)).expect("an override backed by a request input is legal");

    // An override that names no workflow input is a promise the runtime cannot
    // keep, so it is rejected rather than silently ignored.
    let dangling = base.replace("input: request.temperature", "input: request.absent");
    let reported = errors(&dangling);
    assert!(
        reported
            .iter()
            .any(|error| error.contains("request.absent")),
        "{reported:?}"
    );

    // An override backed by an input the caller cannot set is equally a lie.
    let not_request_sourced = base.replace(
        "source: { kind: request }",
        "source: { kind: application, name: request.temperature }",
    );
    let reported = errors(&not_request_sourced);
    assert!(
        reported
            .iter()
            .any(|error| error.contains("request-sourced")),
        "{reported:?}"
    );
}

#[test]
fn constraint_dialect_and_tokenizer_artifact_are_representable() {
    let document = r#"
package:
  tokenizer:
    algorithm: bpe
    vocab_size: 32000
    byte_level: true
    artifacts:
      - location: tokenizer.json
    special_tokens:
      bos: { id: 1, content: "<s>" }
      eos: { id: 2, content: "</s>" }
  constraint_languages:
    - dialect: gbnf
      version: "1"
      component: grammar
pipeline:
  workflow:
    manifest:
      adapter_abis: { onnx-genai.grammar-guidance: "1" }
      capabilities: [workflow_ssa, grammar_guidance_adapter]
    inputs: {}
    components:
      grammar:
        implementation: { kind: adapter, abi: onnx-genai.grammar-guidance, version: "1" }
        ports: {}
    steps:
      - kind: invoke
        component: grammar
"#;
    let metadata = parse(document);
    validate_metadata(&metadata).expect("package facts are valid");
    let package = metadata.package.as_ref().expect("package facts");
    let tokenizer = package.tokenizer.as_ref().expect("tokenizer facts");
    assert_eq!(tokenizer.vocab_size, 32000);
    assert!(tokenizer.byte_level);
    assert_eq!(tokenizer.artifacts[0].location, "tokenizer.json");
    assert_eq!(tokenizer.special_tokens["eos"].content, "</s>");
    assert_eq!(package.constraint_languages[0].dialect, "gbnf");

    // The dialect must name a component that actually parses it.
    let dangling = document.replace("component: grammar\n", "component: absent\n");
    let reported = errors(&dangling);
    assert!(
        reported.iter().any(|error| error.contains("absent")),
        "{reported:?}"
    );
}

#[test]
fn unknown_optional_profiles_are_skippable_and_unknown_core_fields_are_not() {
    // A strict reader must be able to skip a profile it does not understand,
    // without the document as a whole becoming unreadable.
    let ignorable = r#"
profiles:
  future:
    kind: some.future.task
    version: "3"
    requirement: ignorable
"#;
    validate_metadata(&parse(ignorable)).expect("an ignorable unknown profile is skippable");

    // A required profile it does not understand is a hard stop.
    let required = ignorable.replace("requirement: ignorable", "requirement: required");
    let reported = errors(&required);
    assert!(
        reported
            .iter()
            .any(|error| error.contains("some.future.task")),
        "{reported:?}"
    );

    // Unknown *core* fields still fail closed regardless of profile handling.
    assert!(
        serde_yaml::from_str::<InferenceMetadata>("some_future_core_section: {}\n").is_err(),
        "unknown core fields must fail closed"
    );

    // And a skipped profile must not change the semantic identity, or a strict
    // reader and a permissive one would disagree about plan compatibility.
    let without = "schema_version: v1\n";
    let with = format!("schema_version: v1{ignorable}");
    assert_eq!(
        semantic_identity_of_str(without).expect("identity"),
        semantic_identity_of_str(&with).expect("identity")
    );
}

#[test]
fn semantic_identity_tracks_meaning_not_formatting() {
    let yaml = "schema_version: v1\nmodel:\n  name: m\n";
    let json = r#"{"model": {"name": "m"}, "schema_version": "v1"}"#;
    assert_eq!(
        semantic_identity_of_str(yaml).expect("identity"),
        semantic_identity_of_str(json).expect("identity"),
        "encoding and key order must not change plan compatibility"
    );

    let changed = "schema_version: v1\nmodel:\n  name: n\n";
    assert_ne!(
        semantic_identity_of_str(yaml).expect("identity"),
        semantic_identity_of_str(changed).expect("identity"),
        "a semantic change must invalidate a disposable plan"
    );

    // It is an identity, not a signature: the scheme is named so a reader can
    // tell what it is and cannot mistake it for integrity.
    assert!(
        semantic_identity_of_str(yaml)
            .expect("identity")
            .starts_with("onnx-genai-metadata-identity-v1:sha256:")
    );
}

#[test]
fn session_scope_and_release_boundaries_remain_normative() {
    let document = r#"
pipeline:
  workflow:
    manifest:
      capabilities: [workflow_ssa, session_state, session_state_lease, serving_service_contract]
    inputs:
      seed_state:
        contract: { dtype: float32, rank: 2, shape: [batch, hidden], batch_layout: { kind: request_aligned, axis: 0 } }
        role: { kind: opaque }
        source: { kind: application, name: seed_state }
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
    state:
      conversation:
        contract: { dtype: float32, rank: 2, shape: [batch, hidden], batch_layout: { kind: request_aligned, axis: 0 } }
        class: semantic
        scope: session
        initializer: seed_state
        recurrence: { kind: invariant }
        management: runtime
        release_boundary: session
        service_group: conversation_state
    components:
      decoder:
        implementation: { kind: onnx, artifact: decoder.onnx }
        ports:
          inputs:
            past_state: { dtype: float32, rank: 2, shape: [batch, hidden], batch_layout: { kind: request_aligned, axis: 0 } }
          outputs:
            next_state: { dtype: float32, rank: 2, shape: [batch, hidden], batch_layout: { kind: request_aligned, axis: 0 } }
    serving:
      active: active
      done: done
      accepted_len: accepted_len
      state_service:
        groups:
          conversation_state:
            kind: recurrent
            layout: batch_hidden
            ports:
              decoder:
                conversation:
                  input: past_state
                  output: next_state
                  access: read_write
    steps:
      - kind: invoke
        component: decoder
        inputs: { past_state: seed_state }
        outputs: { next_state: decoder.next_state }
"#;
    let metadata = parse(document);
    validate_metadata(&metadata).expect("session-scoped state is declarable");
    let cell = &metadata.pipeline.as_ref().expect("pipeline").workflow.state["conversation"];
    assert_eq!(cell.scope, onnx_genai_metadata::WorkflowStateScope::Session);
    assert_eq!(
        cell.release_boundary,
        Some(onnx_genai_metadata::StateReleaseBoundary::Session)
    );
    assert_eq!(
        cell.management,
        onnx_genai_metadata::StateManagement::Runtime
    );

    // A lease is only normative if something carries it. Here that is the state
    // service group whose alias names the ports the graph reads and writes.
    let workflow = &metadata.pipeline.as_ref().expect("pipeline").workflow;
    let facts = onnx_genai_metadata::classify_session_state(workflow);
    assert_eq!(
        facts.carrier("conversation"),
        Some(onnx_genai_metadata::SessionStateCarrier::StateServiceGroup)
    );
    assert!(facts.carries_any());
    assert_eq!(facts.uncarried().count(), 0);

    // Drop the group and the lease has no reader: session scope alone never
    // said how the next invocation reaches what was kept.
    let unheld = document.replace("        service_group: conversation_state\n", "");
    let reported = errors(&unheld);
    assert!(
        reported
            .iter()
            .any(|error| error.contains("no state service group holds it")),
        "{reported:?}"
    );

    // Runtime-managed state must say when it may be released; otherwise nothing
    // in the document says how long the runtime has to keep it alive.
    let unbounded = document.replace("        release_boundary: session\n", "");
    let reported = errors(&unbounded);
    assert!(
        reported
            .iter()
            .any(|error| error.contains("release_boundary")),
        "{reported:?}"
    );
}

#[test]
fn row_scoped_native_components_declare_their_row_axis() {
    // Row identities are never serialized, so the only thing that makes
    // compaction possible for a stateful native component is a declared row
    // scope. A component with request-aligned ports and per-row state that does
    // not declare one is rejected.
    let metadata = parse(MULTIMODAL_ADAPTER_WORKFLOW);
    let scope = metadata
        .pipeline
        .as_ref()
        .expect("pipeline")
        .workflow
        .components["grammar"]
        .row_scope
        .as_ref()
        .expect("row scope");
    assert_eq!(scope.axis, 0);
    assert!(scope.stateful);

    let undeclared =
        MULTIMODAL_ADAPTER_WORKFLOW.replace("        row_scope: { axis: 0, stateful: true }\n", "");
    let reported = errors(&undeclared);
    assert!(
        reported.iter().any(|error| error.contains("row_scope")),
        "{reported:?}"
    );
}

/// Rubber-duck item 1: no row identity is serialized, yet the row axis stays
/// derivable for every value the runtime has to permute.
#[test]
fn the_row_axis_is_derivable_without_any_serialized_row_identity() {
    let metadata = parse(MULTIMODAL_ADAPTER_WORKFLOW);
    let workflow = &metadata.pipeline.as_ref().expect("pipeline").workflow;

    // Every request-scoped input names the axis that compaction permutes, and
    // none of them name a request, a slot, or an epoch.
    let request_axes = workflow
        .inputs
        .values()
        .filter_map(|input| input.contract.batch_layout.request_axis())
        .collect::<Vec<_>>();
    assert!(
        !request_axes.is_empty(),
        "no request-aligned inputs to compact"
    );
    assert!(
        request_axes.iter().all(|axis| *axis == 0),
        "{request_axes:?}"
    );

    // Shared values are explicitly invariant, so the runtime knows not to
    // permute them rather than having to guess from a name.
    assert!(
        workflow
            .inputs
            .values()
            .any(|input| input.contract.batch_layout.is_shared()),
        "a workflow with no invariant value cannot demonstrate the distinction"
    );
    assert!(
        request_axes.len()
            < workflow
                .inputs
                .values()
                .filter(|input| !input.contract.batch_layout.is_shared())
                .count()
                + request_axes.len(),
        "the layout vocabulary must distinguish more than one kind"
    );

    // The serialized identity vocabulary is gone: adding any of it back is a
    // hard parse error, not a warning.
    for field in [
        "    serving:\n      slot_ids: request.slot_ids\n",
        "    outputs:\n      tokens:\n        row_ids: slot_ids\n",
    ] {
        let document =
            MULTIMODAL_ADAPTER_WORKFLOW.replace("    steps:", &format!("{field}    steps:"));
        assert!(
            serde_yaml::from_str::<InferenceMetadata>(&document).is_err(),
            "row identity must fail closed, not be ignored"
        );
    }
}

/// Rubber-duck item 2: LoRA selection, native grammar state, and vision state
/// all survive compaction, because all three are declared row-scoped on the
/// same axis rather than being special-cased by the runtime.
#[test]
fn every_row_scoped_carrier_survives_batch_compaction() {
    let metadata = parse(MULTIMODAL_ADAPTER_WORKFLOW);
    let workflow = &metadata.pipeline.as_ref().expect("pipeline").workflow;

    // The native grammar component holds per-request parser state and says so.
    let grammar = &workflow.components["grammar"];
    let row_scope = grammar
        .row_scope
        .as_ref()
        .expect("grammar declares row scope");
    assert_eq!(row_scope.axis, 0);
    assert!(row_scope.stateful);

    // LoRA selection is request-aligned, so it permutes with everything else.
    let adapters = metadata.adapters.as_ref().expect("adapters");
    let selection = &adapters.selection;
    for input in [&selection.segments, &selection.scales] {
        let contract = &workflow.inputs[input.as_str()].contract;
        assert_eq!(
            contract.batch_layout.request_axis(),
            Some(0),
            "adapter selection input '{input}' must compact with its row"
        );
    }

    // The vision result is packed across requests but carries an owner mapping,
    // which is what lets packed items follow their row through compaction even
    // though the encoder batched on a different axis than the decoder.
    let vision = &workflow.inputs["image_features"].contract;
    assert!(
        !vision.batch_layout.is_shared(),
        "a packed encoder result is not invariant across requests"
    );
    assert!(
        vision.batch_layout.is_row_scoped(),
        "packed encoder values must still be attributable to a row"
    );
    match &vision.batch_layout {
        onnx_genai_metadata::BatchLayout::TokenPacked { axis, levels, .. } => {
            assert_eq!(*axis, 0);
            assert_eq!(
                levels.len(),
                1,
                "a packing with no coarser grouping owns straight into request rows"
            );
            assert_eq!(levels[0].offsets, "image_offsets");
            assert_eq!(levels[0].owner, "image_owner");
        }
        other => panic!("packed encoder value must declare its owner mapping: {other:?}"),
    }
    assert!(
        workflow.inputs["image_features"].externally_suppliable,
        "an encoder result the runtime may cache or precompute must say so"
    );
}

/// Rubber-duck item 13: a portable checkpoint and a private disaggregated
/// transfer are different mechanisms, and the document says which one a group
/// supports.
#[test]
fn portable_checkpoints_are_distinct_from_private_state_transfer() {
    // A group with no checkpoint adapter is private: its state may still move
    // between processes over a private protocol, but it is not a package output.
    let private = serving_workflow("permitted", SOUND_CAPABILITIES, "", "");
    validate_metadata(&parse(&private)).expect("private state needs no checkpoint adapter");

    let group = |document: &str| {
        parse(document)
            .pipeline
            .expect("pipeline")
            .workflow
            .serving
            .expect("serving")
            .state_service
            .groups
            .remove("decoder_cache")
            .expect("group")
    };
    assert!(
        group(&private).checkpoint.is_none(),
        "silence must mean private, never portable"
    );

    // Publishing that private state as a package output is rejected: portability
    // is a declared property, not an emergent one.
    let exported = private.replace(
        "    state:\n      cache:",
        "    outputs:\n      cache:\n        contract: { dtype: float16, rank: 4, shape: [batch, \
         heads, sequence, head_dim], batch_layout: { kind: request_aligned, axis: 0 } }\n        \
         role: tensor\n        stage: pre_adapter\n    state:\n      cache:",
    );
    let failures = errors(&exported);
    assert!(
        failures
            .iter()
            .any(|error| error.contains("checkpoint adapter")),
        "{failures:?}"
    );

    // Declaring the versioned adapter is what makes the export legal.
    let portable = exported.replace(
        "            capabilities:",
        "            checkpoint: { adapter: onnx-genai.kv-checkpoint, version: \"1\" }\n            \
         capabilities:",
    );
    validate_metadata(&parse(&portable)).expect("a declared checkpoint adapter permits export");
    let checkpoint = group(&portable).checkpoint.expect("checkpoint");
    assert_eq!(checkpoint.adapter, "onnx-genai.kv-checkpoint");
    assert_eq!(checkpoint.version, "1");
}

#[test]
fn the_speculative_region_covers_every_component_in_the_loop_body() {
    // The region is the loop body, not the two named roles. A grammar sidecar
    // invoked between the proposer and the target runs on every speculated
    // position too, so an unrewindable effect there is just as fatal — and
    // naming only the proposer and target would miss it entirely.
    let workflow = serving_workflow("permitted", SOUND_CAPABILITIES, "", &speculative_block(4));
    let with_sidecar = workflow
        .replace(
            "capabilities: [workflow_ssa, serving_service_contract]",
            "capabilities: [workflow_ssa, serving_service_contract, linear_effects, \
             nested_control_flow]",
        )
        .replace(
            r#"      verifier:
        implementation: { kind: onnx, artifact: verifier.onnx }"#,
            r#"      sidecar:
        implementation: { kind: binding }
        ports: {}
        effects: [grammar]
      verifier:
        implementation: { kind: onnx, artifact: verifier.onnx }"#,
        )
        .replace(
            r#"    steps:
      - kind: invoke
        component: proposer
      - kind: invoke
        component: verifier"#,
            r#"    steps:
      - kind: loop
        continue_when: more
        max_iterations: budget
        steps:
          - kind: invoke
            component: proposer
          - kind: invoke
            component: sidecar
          - kind: invoke
            component: verifier"#,
        )
        .replace(
            "      empty_cache:",
            r#"      more:
        contract: { dtype: bool, rank: 0, shape: [] }
        role: { kind: opaque }
        source: { kind: application, name: more }
      budget:
        contract: { dtype: int64, rank: 0, shape: [] }
        role: { kind: opaque }
        source: { kind: application, name: budget }
      empty_cache:"#,
        );

    let unrewindable = with_sidecar.replace(
        "    components:",
        r#"    effects:
      grammar:
        retry: idempotent
        speculation_safety: { kind: none }
    components:"#,
    );
    let reported = errors(&unrewindable);
    assert!(
        reported
            .iter()
            .any(|error| error.contains("speculation_safety none")),
        "an unrewindable effect on a third component in the loop body must be rejected: \
         {reported:?}"
    );

    // The same sidecar with a rewindable bound is accepted.
    let rewindable = with_sidecar.replace("    components:", REWINDABLE_GRAMMAR);
    validate_metadata(&parse(&rewindable)).expect("a rewindable sidecar is speculative-safe");
}

#[test]
fn runtime_owned_state_cannot_be_exported_under_an_alias() {
    // An emit names an SSA value and an output key that need not match. Keying
    // the checkpoint rule off the output name would let a producer publish
    // runtime-owned state as `cache_dump` and never trip the gate, so the rule
    // has to read the emitted value.
    let aliased = serving_workflow("permitted", SOUND_CAPABILITIES, "", "")
        .replace(
            "capabilities: [workflow_ssa, serving_service_contract]",
            "capabilities: [workflow_ssa, serving_service_contract, linear_effects, \
             nested_control_flow, typed_emit]",
        )
        .replace(
            "      empty_cache:",
            r#"      more:
        contract: { dtype: bool, rank: 0, shape: [] }
        role: { kind: opaque }
        source: { kind: application, name: more }
      budget:
        contract: { dtype: int64, rank: 0, shape: [] }
        role: { kind: opaque }
        source: { kind: application, name: budget }
      empty_cache:"#,
        )
        .replace(
            r#"      verifier:
        implementation: { kind: onnx, artifact: verifier.onnx }
        ports: {}"#,
            r#"      verifier:
        implementation: { kind: onnx, artifact: verifier.onnx }
        ports:
          inputs:
            past_key_values: { dtype: float16, rank: 4, shape: [batch, heads, sequence, head_dim], batch_layout: { kind: request_aligned, axis: 0 } }
          outputs:
            present_key_values: { dtype: float16, rank: 4, shape: [batch, heads, sequence, head_dim], batch_layout: { kind: request_aligned, axis: 0 } }"#,
        )
        .replace(
            "    state:\n      cache:",
            r#"    outputs:
      cache_dump:
        contract: { dtype: float16, rank: 4, shape: [batch, heads, sequence, head_dim], batch_layout: { kind: request_aligned, axis: 0 } }
        role: tensor
        stage: pre_adapter
    state:
      cache:"#,
        )
        .replace(
            r#"      - kind: invoke
        component: proposer
      - kind: invoke
        component: verifier"#,
            r#"      - kind: invoke
        component: proposer
      - kind: loop
        continue_when: more
        max_iterations: budget
        carried:
          - cell: cache
            initial: empty_cache
            next: next_cache
        steps:
          - kind: invoke
            component: verifier
            inputs: { past_key_values: cache }
            outputs: { present_key_values: next_cache }
          - kind: emit
            value: cache
            output: cache_dump
            mode: replace"#,
        );
    let reported = errors(&aliased);
    assert!(
        reported
            .iter()
            .any(|error| error.contains("checkpoint adapter")),
        "exporting runtime-owned state under an alias must be rejected: {reported:?}"
    );

    // Declaring the versioned adapter is what makes the aliased export legal.
    let portable = aliased.replace(
        "            capabilities:",
        "            checkpoint: { adapter: onnx-genai.kv-checkpoint, version: \"1\" }\n            \
         capabilities:",
    );
    validate_metadata(&parse(&portable)).expect("a declared checkpoint adapter permits export");
}

#[test]
fn semantic_identity_ignores_how_a_number_was_spelled() {
    // A float field accepts `1` and `1.0` for the same value, and serde_json
    // remembers which one it parsed. If the identity remembered too, a cosmetic
    // rewrite would invalidate every compiled plan and checkpoint keyed off it.
    let base = serving_workflow("permitted", SOUND_CAPABILITIES, "", "");
    let integral = format!("{base}generation:\n  defaults:\n    temperature: 1\n");
    let decimal = format!("{base}generation:\n  defaults:\n    temperature: 1.0\n");
    assert_ne!(integral, decimal, "the two spellings must differ as text");
    validate_metadata(&parse(&integral)).expect("an integral spelling parses");
    validate_metadata(&parse(&decimal)).expect("a decimal spelling parses");
    assert_eq!(
        semantic_identity_of_str(&integral).expect("identity"),
        semantic_identity_of_str(&decimal).expect("identity"),
        "numeric spelling is formatting, not meaning"
    );

    // Meaning still moves the identity: a different temperature is a different
    // sampling contract, and any plan keyed off the old one must be discarded.
    let warmer = format!("{base}generation:\n  defaults:\n    temperature: 1.5\n");
    assert_ne!(
        semantic_identity_of_str(&integral).expect("identity"),
        semantic_identity_of_str(&warmer).expect("identity"),
        "a different temperature is a different contract"
    );
}

/// An encoder-conditioned audio workflow: encoded bytes enter as a request
/// input, a declarative audio program turns them into a feature tensor, and the
/// encoder consumes that tensor. The program is data — no model-family name and
/// no runtime branch appears anywhere in the document.
const AUDIO_PREPROCESSING_WORKFLOW: &str = r#"
schema_version: v1
preprocessing:
  audio:
    transforms:
      - op: decode
        outputs: [samples]
      - op: resample
        inputs: [samples]
        outputs: [resampled]
        sample_rate: 16000
      - op: pad
        inputs: [resampled]
        outputs: [windowed]
        mode: fixed_window
        target_length: 480000
        pad_value: 0.0
      - op: log_mel
        inputs: [windowed]
        outputs: [mel]
        num_mel_bins: 80
        n_fft: 400
        hop_length: 160
        window: hann
        mel_scale: slaney
        sample_rate: 16000
      - op: normalize
        inputs: [mel]
        outputs: [features]
        mode: whisper_log_mel
    outputs:
      - source: features
        name: audio.input_features
        content: audio_features
        dtype: float32
        contract:
          dtype: float32
          rank: 3
          shape: [batch, 80, audio_seq_len]
          batch_layout: { kind: request_aligned, axis: 0 }
pipeline:
  workflow:
    manifest:
      adapter_abis: { onnx-genai.audio-preprocess: "1" }
      capabilities: [workflow_ssa, typed_emit, audio_preprocessing_program]
    inputs:
      request.audio:
        contract: { dtype: uint8, rank: 1, shape: [encoded_bytes] }
        role: { kind: opaque }
        source: { kind: application, name: audio }
    outputs:
      encoder_states:
        contract: { dtype: float32, rank: 3, shape: [batch, 1500, 384], batch_layout: { kind: request_aligned, axis: 0 } }
        role: tensor
        stage: pre_adapter
    components:
      audio_preprocess:
        implementation: { kind: adapter, abi: onnx-genai.audio-preprocess, version: "1" }
        ports:
          inputs:
            encoded: { dtype: uint8, rank: 1, shape: [encoded_bytes] }
          outputs:
            input_features: { dtype: float32, rank: 3, shape: [batch, 80, audio_seq_len], batch_layout: { kind: request_aligned, axis: 0 } }
      encoder:
        implementation: { kind: onnx, artifact: encoder.onnx }
        ports:
          inputs:
            input_features: { dtype: float32, rank: 3, shape: [batch, 80, audio_seq_len], batch_layout: { kind: request_aligned, axis: 0 } }
          outputs:
            encoder_hidden_states: { dtype: float32, rank: 3, shape: [batch, 1500, 384], batch_layout: { kind: request_aligned, axis: 0 } }
    steps:
      - kind: invoke
        component: audio_preprocess
        inputs: { encoded: request.audio }
        outputs: { input_features: audio.input_features }
      - kind: invoke
        component: encoder
        inputs: { input_features: audio.input_features }
        outputs: { encoder_hidden_states: states }
      - kind: emit
        value: states
        output: encoder_states
        mode: replace
"#;

#[test]
fn a_declarative_audio_program_is_expressible_in_a_workflow() {
    validate_metadata(&parse(AUDIO_PREPROCESSING_WORKFLOW))
        .expect("a typed audio preprocessing program is a valid workflow document");
}

#[test]
fn an_audio_adapter_without_a_program_is_rejected() {
    let document = AUDIO_PREPROCESSING_WORKFLOW
        .split("pipeline:")
        .nth(1)
        .map(|rest| format!("schema_version: v1\npipeline:{rest}"))
        .expect("the fixture declares a pipeline");

    let errors = errors(&document);

    assert!(
        errors
            .iter()
            .any(|error| error.contains("require preprocessing.audio metadata")),
        "{errors:?}"
    );
}

#[test]
fn an_audio_output_without_a_contract_is_rejected() {
    let document = AUDIO_PREPROCESSING_WORKFLOW.replace(
        "        contract:\n          dtype: float32\n          rank: 3\n          shape: \
         [batch, 80, audio_seq_len]\n          batch_layout: { kind: request_aligned, axis: 0 }\n",
        "",
    );

    let errors = errors(&document);

    assert!(
        errors
            .iter()
            .any(|error| error.contains("must declare a TensorContract")),
        "{errors:?}"
    );
}

#[test]
fn an_audio_output_that_no_invocation_produces_is_rejected() {
    let document =
        AUDIO_PREPROCESSING_WORKFLOW.replace("name: audio.input_features", "name: audio.unbound");

    let errors = errors(&document);

    assert!(
        errors
            .iter()
            .any(|error| error.contains("must be a declared SSA output")),
        "{errors:?}"
    );
}

#[test]
fn an_audio_feature_tensor_must_stay_request_aligned() {
    let document = AUDIO_PREPROCESSING_WORKFLOW.replace(
        "          batch_layout: { kind: request_aligned, axis: 0 }\n",
        "          batch_layout: { kind: shared }\n",
    );

    let errors = errors(&document);

    // A shared feature tensor would let encoder states drift away from the
    // decoder rows that consume them, so the contract mismatch must fail closed.
    assert!(
        errors
            .iter()
            .any(|error| error.contains("preprocessing.audio output 'audio.input_features'")),
        "{errors:?}"
    );
}

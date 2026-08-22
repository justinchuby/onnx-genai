//! Gemma 4 E2B target + assistant: the canonical metadata expresses the whole
//! contract generically, and the resolved decode ABI proves exact state
//! ownership without any model-name branch.
//!
//! These two catalogue examples are config-only, so this file proves *contract*
//! behavior — the facts a runtime reads before it executes a graph — rather than
//! numerical output. It asserts:
//!
//!   * the target owns its hybrid full/sliding KV (read-write), with
//!     heterogeneous global/local head widths, and is dense (no MoE invented);
//!   * the assistant owns NO KV (its shared full/sliding aliases are
//!     `read_only`, so the resolved ABI is `kv_ownership: shared` with no KV
//!     transitions), and its only loop-carried cell is `projected_state`;
//!   * the speculative contract is a chained proposal over a pruned (`subset`)
//!     vocabulary that borrows the target's ordered embedding table, and every
//!     rewound cell is covered to the maximum proposal width.

use onnx_genai_metadata::{
    InferenceMetadata, KvOwnership, SpeculativeProposalExecution, SpeculativeVocabulary, StateKind,
    StatePortAccess, compile_workflow, decoder_abi, validate_metadata,
};

const TARGET: &str =
    include_str!("../../../examples/inference_metadata/catalogue/23-gemma4-e2b-decoder.yaml");
const ASSISTANT: &str = include_str!(
    "../../../examples/inference_metadata/catalogue/24-gemma4-e2b-assistant-speculative.yaml"
);
const QWEN3: &str = include_str!(
    "../../../examples/inference_metadata/catalogue/22-qwen3-chained-speculative-decoding.yaml"
);

fn parse(doc: &str) -> InferenceMetadata {
    let metadata = serde_yaml::from_str::<InferenceMetadata>(doc).expect("example parses");
    validate_metadata(&metadata).expect("example validates");
    metadata
}

/// The target decoder owns its hybrid attention KV with independent global and
/// local head widths. This checkpoint is dense, so no MoE metadata is invented.
#[test]
fn target_owns_hybrid_dense_decoder() {
    let metadata = parse(TARGET);
    let workflow = &metadata.pipeline.as_ref().expect("pipeline").workflow;

    // The resolved decode ABI shows the decoder owns every KV transition.
    let abi = decoder_abi(workflow, "decoder").expect("decoder ABI resolves");
    assert_eq!(abi.kv_ownership, Some(KvOwnership::Owned));
    let kv_inputs = abi.kv_inputs.as_ref().expect("decoder owns KV inputs");
    // Two global-layer owners (K+V each) plus one local-layer owner (K+V) = 6.
    assert_eq!(kv_inputs.len(), 6, "two full owners + one sliding owner");
    assert!(
        abi.state_pairs.is_none(),
        "a plain decoder carries no recurrent state"
    );

    // Hybrid attention is two groups with different discipline: the local window
    // is evictable, the global context is not.
    let groups = &workflow
        .serving
        .as_ref()
        .expect("serving")
        .state_service
        .groups;
    let full = &groups["full_attention"];
    let sliding = &groups["sliding_attention"];
    assert_eq!(full.kind, StateKind::FullAttention);
    assert_eq!(sliding.kind, StateKind::SlidingAttention);
    assert!(
        !full.reuse.evictable_prefix,
        "global context is never evicted"
    );
    assert!(
        sliding.reuse.evictable_prefix,
        "the local window drops old tokens"
    );

    // Heterogeneous global/local geometry: this checkpoint's real heterogeneity
    // is head WIDTH — the global head_dim differs from the local head_dim. The
    // dimensions are independent symbolic axes on the graph ports.
    let dim = |cell: &str, axis: usize| {
        workflow.state[cell].contract.shape.as_ref().expect("shape")[axis].clone()
    };
    assert_ne!(
        dim("full_key_0", 3),
        dim("sliding_key_0", 3),
        "global vs local head width"
    );

    // Fewer physical KV owners than logical attention layers: the full group
    // names two owner layers, the sliding group one; borrowing layers expose no
    // ports at all.
    let layers: std::collections::BTreeSet<usize> = full.ports["decoder"]
        .values()
        .filter_map(|alias| alias.layer)
        .collect();
    assert_eq!(layers, std::collections::BTreeSet::from([0, 1]));

    // The shipping E2B checkpoint is dense: no MoE metadata is invented. (The
    // schema still expresses a sparse FFN through `model.mixture_of_experts`
    // for MoE variants; this checkpoint declares none.)
    assert!(
        metadata
            .model
            .as_ref()
            .and_then(|model| model.mixture_of_experts.as_ref())
            .is_none(),
        "the E2B target checkpoint is dense"
    );
}

/// The assistant reads the target's KV read-only and carries only its projected
/// state, so the resolved ABI proves it owns no cache at all.
#[test]
fn assistant_is_cacheless_read_only_reader() {
    let metadata = parse(ASSISTANT);
    let workflow = &metadata.pipeline.as_ref().expect("pipeline").workflow;

    // The target half still owns its KV.
    let target_abi = decoder_abi(workflow, "target").expect("target ABI resolves");
    assert_eq!(target_abi.kv_ownership, Some(KvOwnership::Owned));

    // The assistant half owns nothing at all: every full/sliding alias it holds
    // is read-only (no KV transitions, shared ownership), and its carry is
    // folded into the fused inputs_embeds rather than a recurrent state cell —
    // so the resolved ABI has no state pairs either.
    let assistant_abi = decoder_abi(workflow, "assistant").expect("assistant ABI resolves");
    assert_eq!(assistant_abi.kv_ownership, Some(KvOwnership::Shared));
    assert!(
        assistant_abi.kv_inputs.is_none(),
        "read-only shares are not KV transitions"
    );
    assert!(
        assistant_abi.state_pairs.is_none(),
        "a folded carry is not a recurrent state cell"
    );

    // The same state groups carry a read-write target alias and a read-only
    // assistant alias for the very same cells: one advances the prefix, the
    // other only observes it.
    let groups = &workflow
        .serving
        .as_ref()
        .expect("serving")
        .state_service
        .groups;
    for group in ["full_attention", "sliding_attention"] {
        let ports = &groups[group].ports;
        for alias in ports["target"].values() {
            assert_eq!(
                alias.access,
                StatePortAccess::ReadWrite,
                "target advances {group}"
            );
        }
        for alias in ports["assistant"].values() {
            assert_eq!(
                alias.access,
                StatePortAccess::ReadOnly,
                "assistant only reads {group}"
            );
        }
    }

    // The assistant reads a single merged representative of each group, so it
    // binds fewer aliases than the target owns.
    let full_target_ports = groups["full_attention"].ports["target"].len();
    let full_assistant_ports = groups["full_attention"].ports["assistant"].len();
    assert!(
        full_assistant_ports < full_target_ports,
        "assistant borrows a single merged representative"
    );
    // The merged borrow carries no layer index (it maps to no specific owner).
    for alias in groups["full_attention"].ports["assistant"].values() {
        assert!(
            alias.layer.is_none(),
            "a merged representative carries no layer"
        );
        assert!(
            alias.output.is_none(),
            "a pure reader exposes no present output"
        );
    }
}

/// The speculative wiring is a chained proposal over a pruned vocabulary that
/// borrows the target's ordered embeddings, with every rewound cell covered.
#[test]
fn assistant_speculative_contract_is_chained_pruned_and_rewindable() {
    let metadata = parse(ASSISTANT);
    let speculative = metadata.speculative.as_ref().expect("speculative contract");
    assert_eq!(speculative.proposer, "assistant");
    assert_eq!(speculative.target, "target");

    // Chained proposal with a FOLDED carry: the drafter emits its next carry as
    // an output that re-enters through the fused inputs_embeds, so there is no
    // separate recurrent binding.
    match &speculative.proposal_execution {
        SpeculativeProposalExecution::Chained {
            token_embedding_input,
            logits_output,
            recurrent,
            folded_carry_output,
            ..
        } => {
            assert_eq!(token_embedding_input, "inputs_embeds");
            assert_eq!(logits_output, "draft_logits");
            assert!(
                recurrent.is_empty(),
                "the carry is folded, not a separate port"
            );
            assert_eq!(folded_carry_output.as_deref(), Some("next_projected_state"));
        }
        other => panic!("expected a chained proposal, got {other:?}"),
    }

    // Concatenated target hidden handoff. The tied target embedding is
    // graph-internal, so no external shared_weights file is referenced.
    assert_eq!(
        speculative
            .port_bindings
            .get("target_hidden_context")
            .map(String::as_str),
        Some("inputs_embeds")
    );
    assert!(
        speculative.shared_weights.is_empty(),
        "the tied target embedding is graph-internal, not externally shared"
    );

    // Read-only shared attention groups, not owned draft caches.
    assert!(speculative.shared_state.contains("full_attention"));
    assert!(speculative.shared_state.contains("sliding_attention"));

    // Full-vocab drafter: the centroid pruning is graph-internal, so the
    // vocabulary relationship is identical (the drafter emits the target axis).
    match &speculative.vocabulary {
        SpeculativeVocabulary::Identical => {}
        other => panic!("expected an identical vocabulary, got {other:?}"),
    }
    assert!(
        speculative.distribution_preserving,
        "standard rejection sampling preserves the distribution despite the pruned head"
    );

    // Every rewound cell is bound to a group covered to the proposal width.
    let workflow = &metadata.pipeline.as_ref().expect("pipeline").workflow;
    let groups = &workflow
        .serving
        .as_ref()
        .expect("serving")
        .state_service
        .groups;
    for cell in &speculative.rollback_state {
        let group_name = workflow.state[cell]
            .service_group
            .as_ref()
            .unwrap_or_else(|| panic!("rollback cell {cell} binds a group"));
        let positions = groups[group_name]
            .capabilities
            .rollback_positions
            .unwrap_or_else(|| panic!("group {group_name} declares rollback_positions"));
        assert!(
            positions >= speculative.max_proposal_width,
            "group {group_name} covers the proposal width"
        );
    }
}

/// Both packages lower to an executable workflow plan, and the lowering tracks
/// every state cell — the runtime step the engine performs before it binds ORT.
#[test]
fn both_packages_lower_to_an_executable_plan() {
    for (label, doc, state_domain) in [
        ("target", TARGET, "state:full_key_0"),
        ("assistant", ASSISTANT, "state:sliding_key_0"),
    ] {
        let metadata = parse(doc);
        let workflow = &metadata.pipeline.as_ref().expect("pipeline").workflow;
        let compiled = compile_workflow(workflow)
            .unwrap_or_else(|error| panic!("{label} did not lower: {error}"));
        assert!(
            compiled.initial_effects.contains_key(state_domain),
            "{label} lowering must track {state_domain}"
        );
    }
}

/// The read-only relaxation is narrow: a read-WRITE alias that names no output
/// port is still rejected, because a transition with no output cannot be
/// written back.
#[test]
fn a_read_write_alias_still_requires_an_output() {
    // Flip one of the assistant's read-only borrows into a (default) read-write
    // alias while leaving it without an output port.
    let mutated = ASSISTANT.replace(
        "full_key_1: {input: shared_kv.full_attention.key, access: read_only, role: key}",
        "full_key_1: {input: shared_kv.full_attention.key, role: key}",
    );
    assert_ne!(
        mutated, ASSISTANT,
        "the read-only alias line must be present to mutate"
    );
    let metadata = serde_yaml::from_str::<InferenceMetadata>(&mutated).expect("mutated parses");
    let errors = validate_metadata(&metadata).expect_err("a read-write alias with no output fails");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("declares no output")),
        "expected a missing-output error, got: {errors:?}"
    );
}

/// The folded-carry contract fails closed: a chained proposer that declares
/// neither a `recurrent` binding nor a `folded_carry_output` is rejected.
#[test]
fn a_chained_proposal_needs_a_recurrent_or_folded_carry() {
    let mutated = ASSISTANT.replace("    folded_carry_output: next_projected_state\n", "");
    assert_ne!(
        mutated, ASSISTANT,
        "the folded_carry_output line must be present to mutate"
    );
    let metadata = serde_yaml::from_str::<InferenceMetadata>(&mutated).expect("mutated parses");
    let errors =
        validate_metadata(&metadata).expect_err("a chained proposal with no carry must fail");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("at least one recurrent binding or a folded_carry_output")),
        "expected a missing-carry error, got: {errors:?}"
    );
}

/// The folded-carry output must be a real proposer output port.
#[test]
fn a_folded_carry_output_must_be_a_proposer_output() {
    let mutated = ASSISTANT.replace(
        "folded_carry_output: next_projected_state",
        "folded_carry_output: not_a_real_port",
    );
    assert_ne!(
        mutated, ASSISTANT,
        "the folded_carry_output line must be present to mutate"
    );
    let metadata = serde_yaml::from_str::<InferenceMetadata>(&mutated).expect("mutated parses");
    let errors = validate_metadata(&metadata)
        .expect_err("a folded_carry_output naming a non-output port must fail");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("folded_carry_output")
                && error.contains("not an output port")),
        "expected a folded-output-port error, got: {errors:?}"
    );
}

/// The folded-carry contract fails closed: a proposer that folds a carry back
/// into its fused input MUST name the first-step target context through
/// `port_bindings.target_hidden_context`, or the carry has no seed.
#[test]
fn a_folded_carry_requires_a_target_hidden_context_binding() {
    let mutated = ASSISTANT.replace("    target_hidden_context: inputs_embeds\n", "");
    assert_ne!(
        mutated, ASSISTANT,
        "the target_hidden_context line must be present to remove"
    );
    let metadata = serde_yaml::from_str::<InferenceMetadata>(&mutated).expect("mutated parses");
    let errors = validate_metadata(&metadata)
        .expect_err("a folded carry with no target_hidden_context must fail");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("port_bindings.target_hidden_context")),
        "expected a missing-context error, got: {errors:?}"
    );
}

/// The first-step context must be a real proposer input port: it is where the
/// fused `concat(token_embedding, carry)` re-enters the graph.
#[test]
fn a_target_hidden_context_must_be_a_proposer_input() {
    let mutated = ASSISTANT.replace(
        "target_hidden_context: inputs_embeds",
        "target_hidden_context: not_a_real_input",
    );
    assert_ne!(
        mutated, ASSISTANT,
        "the target_hidden_context line must be present to mutate"
    );
    let metadata = serde_yaml::from_str::<InferenceMetadata>(&mutated).expect("mutated parses");
    let errors = validate_metadata(&metadata)
        .expect_err("a target_hidden_context naming a non-input port must fail");
    assert!(
        errors.iter().any(
            |error| error.contains("port_bindings.target_hidden_context")
                && error.contains("not an input port")
        ),
        "expected a context-input-port error, got: {errors:?}"
    );
}

/// The folded carry's carry_0 source is explicit: a `folded_carry_output`
/// without a `folded_carry_seed` naming the target output that seeds it is
/// rejected, so the runtime never infers "the target hidden output".
#[test]
fn a_folded_carry_requires_an_explicit_seed() {
    let mutated = ASSISTANT.replace(
        "    folded_carry_seed: {component: target, output: hidden}\n",
        "",
    );
    assert_ne!(
        mutated, ASSISTANT,
        "the folded_carry_seed line must be present"
    );
    let metadata = serde_yaml::from_str::<InferenceMetadata>(&mutated).expect("mutated parses");
    let errors = validate_metadata(&metadata).expect_err("a folded carry with no seed must fail");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("folded_carry_seed")),
        "expected a missing-seed error, got: {errors:?}"
    );
}

/// The seed must name a real target output: an unknown component or a port that
/// is not an output of it is rejected.
#[test]
fn a_folded_carry_seed_must_name_a_real_target_output() {
    let unknown_component = ASSISTANT.replace(
        "folded_carry_seed: {component: target, output: hidden}",
        "folded_carry_seed: {component: ghost, output: hidden}",
    );
    assert_ne!(
        unknown_component, ASSISTANT,
        "the seed line must be present"
    );
    let metadata =
        serde_yaml::from_str::<InferenceMetadata>(&unknown_component).expect("mutated parses");
    let errors =
        validate_metadata(&metadata).expect_err("a seed on an unknown component must fail");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("folded_carry_seed component")
                && error.contains("not a declared")),
        "expected an unknown-seed-component error, got: {errors:?}"
    );

    let bad_output = ASSISTANT.replace(
        "folded_carry_seed: {component: target, output: hidden}",
        "folded_carry_seed: {component: target, output: not_an_output}",
    );
    let metadata = serde_yaml::from_str::<InferenceMetadata>(&bad_output).expect("mutated parses");
    let errors = validate_metadata(&metadata).expect_err("a seed naming a non-output must fail");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("folded_carry_seed output")
                && error.contains("not an output port")),
        "expected a seed-output-port error, got: {errors:?}"
    );
}

/// The embedding source is explicit: a `folded_carry_output` without a
/// `token_embedding` naming where `embed(last_token)` is gathered is rejected,
/// so the runtime never extracts an in-model initializer heuristically.
#[test]
fn a_folded_carry_requires_an_explicit_token_embedding() {
    let mutated = ASSISTANT.replace(
        "    token_embedding: {component: target, table: model.embed_tokens.weight}\n",
        "",
    );
    assert_ne!(
        mutated, ASSISTANT,
        "the token_embedding line must be present"
    );
    let metadata = serde_yaml::from_str::<InferenceMetadata>(&mutated).expect("mutated parses");
    let errors =
        validate_metadata(&metadata).expect_err("a folded carry with no token_embedding must fail");
    assert!(
        errors.iter().any(|error| error.contains("token_embedding")),
        "expected a missing-embedding error, got: {errors:?}"
    );
}

/// The embedding source must name a real component.
#[test]
fn a_token_embedding_must_name_a_real_component() {
    let mutated = ASSISTANT.replace(
        "token_embedding: {component: target, table: model.embed_tokens.weight}",
        "token_embedding: {component: ghost, table: model.embed_tokens.weight}",
    );
    assert_ne!(
        mutated, ASSISTANT,
        "the token_embedding line must be present"
    );
    let metadata = serde_yaml::from_str::<InferenceMetadata>(&mutated).expect("mutated parses");
    let errors = validate_metadata(&metadata)
        .expect_err("a token_embedding on an unknown component must fail");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("token_embedding component")
                && error.contains("not a declared")),
        "expected an unknown-embedding-component error, got: {errors:?}"
    );
}

/// carry_0 is the target's OWN per-token hidden output, so a `folded_carry_seed`
/// naming any component other than the speculative target — a *proposer* seed in
/// particular — is rejected fail-closed, even when it names a real output port.
#[test]
fn a_folded_carry_seed_must_come_from_the_target() {
    // `draft_logits` is a genuine proposer (assistant) output, so the output
    // port resolves; the seed is rejected only because it is not the target.
    let mutated = ASSISTANT.replace(
        "folded_carry_seed: {component: target, output: hidden}",
        "folded_carry_seed: {component: assistant, output: draft_logits}",
    );
    assert_ne!(
        mutated, ASSISTANT,
        "the folded_carry_seed line must be present"
    );
    let metadata = serde_yaml::from_str::<InferenceMetadata>(&mutated).expect("mutated parses");
    let errors =
        validate_metadata(&metadata).expect_err("a proposer-sourced folded carry seed must fail");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("folded_carry_seed component")
                && error.contains("must be the speculative target")),
        "expected a non-target-seed error, got: {errors:?}"
    );
}

/// A folded carry re-enters through the fused input's trailing half, so the
/// DESTINATION `port_bindings.target_hidden_context` must equal the fused
/// `token_embedding_input`. A different — even valid — proposer input port is
/// rejected fail-closed, because a folded carry has no separate destination.
#[test]
fn a_folded_carry_target_hidden_context_must_equal_token_embedding_input() {
    // `shared_kv.full_attention.key` is a real assistant input port, but it is
    // not the fused `inputs_embeds` the carry folds into.
    let mutated = ASSISTANT.replace(
        "target_hidden_context: inputs_embeds",
        "target_hidden_context: shared_kv.full_attention.key",
    );
    assert_ne!(
        mutated, ASSISTANT,
        "the target_hidden_context line must be present to mutate"
    );
    let metadata = serde_yaml::from_str::<InferenceMetadata>(&mutated).expect("mutated parses");
    let errors = validate_metadata(&metadata)
        .expect_err("a target_hidden_context that is not the fused input must fail");
    assert!(
        errors.iter().any(
            |error| error.contains("port_bindings.target_hidden_context")
                && error.contains("must equal the fused token_embedding_input")
        ),
        "expected a mismatched-destination error, got: {errors:?}"
    );
}

/// The fused input's leading half is `embed(last_token)` gathered from the
/// TARGET model's embedding, so `token_embedding.table` must resolve to a real
/// initializer in the named target model/artifact. A table attributed to the
/// wrong (proposer) model, and an empty table, are both rejected fail-closed.
#[test]
fn a_folded_carry_token_embedding_must_resolve_to_a_real_target_initializer() {
    // Attributing the target's embedding table to the proposer model cannot
    // resolve to a real initializer in the *target* model/artifact.
    let wrong_model = ASSISTANT.replace(
        "token_embedding: {component: target, table: model.embed_tokens.weight}",
        "token_embedding: {component: assistant, table: model.embed_tokens.weight}",
    );
    assert_ne!(
        wrong_model, ASSISTANT,
        "the token_embedding line must be present"
    );
    let metadata = serde_yaml::from_str::<InferenceMetadata>(&wrong_model).expect("mutated parses");
    let errors = validate_metadata(&metadata)
        .expect_err("a token_embedding table on the wrong model must fail");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("token_embedding component")
                && error.contains("must be the speculative target")),
        "expected a wrong-model-table error, got: {errors:?}"
    );

    // A bogus (empty) table names no initializer at all.
    let empty_table = ASSISTANT.replace(
        "token_embedding: {component: target, table: model.embed_tokens.weight}",
        "token_embedding: {component: target, table: \"\"}",
    );
    assert_ne!(
        empty_table, ASSISTANT,
        "the token_embedding line must be present"
    );
    let metadata = serde_yaml::from_str::<InferenceMetadata>(&empty_table).expect("mutated parses");
    let errors =
        validate_metadata(&metadata).expect_err("a token_embedding with an empty table must fail");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("token_embedding table")
                && error.contains("real target initializer")),
        "expected an empty-table error, got: {errors:?}"
    );
}

/// A recurrence must resolve to a state-service alias the proposer owns: a
/// binding whose cell has no `groups.*.ports.<proposer>` alias is rejected, so
/// a "recurrence" that is never actually carried cannot slip through.
#[test]
fn a_recurrence_needs_a_proposer_state_service_alias() {
    // `target_cache` is a valid rollback cell, but its alias lives on the
    // verifier, not the proposer — the recurrence cannot resolve.
    let mutated = QWEN3.replace("state: draft_cache", "state: target_cache");
    assert_ne!(mutated, QWEN3, "the recurrent binding must be present");
    let metadata = serde_yaml::from_str::<InferenceMetadata>(&mutated).expect("mutated parses");
    let errors =
        validate_metadata(&metadata).expect_err("a recurrence with no proposer alias must fail");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("no state-service alias on proposer")),
        "expected a missing-alias error, got: {errors:?}"
    );
}

/// A recurrence advances the cell, so its alias must be `read_write`: a
/// `read_only` borrow is frozen and could never carry the loop forward.
#[test]
fn a_recurrence_alias_must_be_read_write() {
    let mutated = QWEN3.replace(
        "draft_cache: {input: past_key_values, output: present_key_values}",
        "draft_cache: {input: past_key_values, output: present_key_values, access: read_only}",
    );
    assert_ne!(mutated, QWEN3, "the draft_cache alias must be present");
    let metadata = serde_yaml::from_str::<InferenceMetadata>(&mutated).expect("mutated parses");
    let errors = validate_metadata(&metadata).expect_err("a read_only recurrence alias must fail");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("read_only") && error.contains("read_write")),
        "expected a read-only recurrence error, got: {errors:?}"
    );
}

/// The recurrence and its alias must name the same output port: a mismatch
/// means the runtime would carry a different value than the binding claims.
#[test]
fn a_recurrence_output_must_match_its_alias() {
    let mutated = QWEN3.replace(
        "draft_cache: {input: past_key_values, output: present_key_values}",
        "draft_cache: {input: past_key_values, output: next_hidden}",
    );
    assert_ne!(mutated, QWEN3, "the draft_cache alias must be present");
    let metadata = serde_yaml::from_str::<InferenceMetadata>(&mutated).expect("mutated parses");
    let errors =
        validate_metadata(&metadata).expect_err("a mismatched recurrence output must fail");
    assert!(
        errors.iter().any(|error| error.contains("binds output")
            && error.contains("state-service alias names output")),
        "expected an output-mismatch error, got: {errors:?}"
    );
}

/// Backward-compatible with #1696: the separate-port `recurrent` form (no
/// `folded_carry_output`) still validates — asserted on the checked-in Qwen3
/// chained example.
#[test]
fn recurrent_chained_form_remains_valid() {
    let doc = include_str!(
        "../../../examples/inference_metadata/catalogue/22-qwen3-chained-speculative-decoding.yaml"
    );
    let metadata = serde_yaml::from_str::<InferenceMetadata>(doc).expect("qwen3 chained parses");
    validate_metadata(&metadata).expect("the recurrent chained form still validates");
    let speculative = metadata.speculative.as_ref().expect("speculative");
    match &speculative.proposal_execution {
        SpeculativeProposalExecution::Chained {
            recurrent,
            folded_carry_output,
            ..
        } => {
            assert!(!recurrent.is_empty(), "qwen3 uses separate-port recurrence");
            assert!(
                folded_carry_output.is_none(),
                "qwen3 declares no folded carry"
            );
        }
        other => panic!("expected chained, got {other:?}"),
    }
}

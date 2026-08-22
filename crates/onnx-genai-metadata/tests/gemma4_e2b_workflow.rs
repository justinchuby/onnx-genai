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

    // The assistant reads a subset of the target's physical owners: the target
    // owns two global layers, the assistant reads only layer 0.
    let full_target_layers = groups["full_attention"].ports["target"].len();
    let full_assistant_layers = groups["full_attention"].ports["assistant"].len();
    assert!(
        full_assistant_layers < full_target_layers,
        "assistant reads a smaller owner set"
    );
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

    // Concatenated target hidden handoff and borrowed ordered embeddings.
    assert_eq!(
        speculative
            .port_bindings
            .get("target_hidden_context")
            .map(String::as_str),
        Some("inputs_embeds")
    );
    assert!(speculative.shared_weights.contains("target_embedding.f32"));

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
        !speculative.distribution_preserving,
        "a lossy centroid drafter is opt-in"
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
        "full_key_0: {input: full_key_0, access: read_only, role: key, layer: 0}",
        "full_key_0: {input: full_key_0, role: key, layer: 0}",
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

use onnx_genai_metadata::{
    extensions::{DFLASH_FLAT_BLOCK_V1, find},
    parse_metadata, validate_metadata,
};

fn document(version: &str, contract_version: &str, probabilities: bool) -> String {
    let probability_port = if probabilities {
        r#"
            proposal_probabilities:
              dtype: float32
              shape: [batch, proposal, 13]
              batch_layout: { kind: request_aligned, axis: 0 }"#
    } else {
        ""
    };
    let probability_binding = if probabilities {
        "          proposal_probabilities: proposal_probabilities\n"
    } else {
        ""
    };
    let probability_output = if probabilities {
        "      proposal_probabilities: proposal_probabilities\n"
    } else {
        ""
    };
    format!(
        r#"
schema_version: {version}
package:
  tokenizer:
    special_tokens:
      eos_token_id: [12]
pipeline:
  workflow:
    manifest: {{}}
    inputs:
      request.active:
        contract: {{ dtype: bool, shape: [batch], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: active }}
        required: true
      request.done:
        contract: {{ dtype: bool, shape: [batch], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: done }}
        required: true
      request.accepted_len:
        contract: {{ dtype: int64, shape: [batch], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: accepted_len }}
        required: true
      request.target_tokens:
        contract: {{ dtype: int64, shape: [batch, verify], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: runtime, version: v1, role: prompt_tokens }}
        source: {{ kind: request }}
        required: true
      request.target_cache:
        contract: {{ dtype: float32, shape: [batch, state_sequence, 2], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: target_cache }}
        required: true
      request.draft_cache:
        contract: {{ dtype: float32, shape: [batch, state_sequence, 2], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: draft_cache }}
        required: true
      request.target_hidden:
        contract: {{ dtype: float32, shape: [batch, sequence, 4], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: target_hidden }}
        required: true
      request.noise:
        contract: {{ dtype: float32, shape: [batch, block, 2], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: noise }}
        required: true
      request.masked:
        contract: {{ dtype: bool, shape: [batch, block], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: masked }}
        required: true
      request.positions:
        contract: {{ dtype: int64, shape: [batch, total], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: positions }}
        required: true
      request.attention:
        contract: {{ dtype: int64, shape: [batch, total], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: attention }}
        required: true
      request.output_projection:
        contract: {{ dtype: float32, shape: [2, 13], batch_layout: {{ kind: shared }} }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: output_projection }}
        required: true
    outputs:
      target_logits:
        contract: {{ dtype: float32, shape: [batch, verify, 13], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: tensor
        family: {{ kind: materialized }}
        stage: pre_adapter
    components:
      termination_policy:
        implementation: {{ kind: binding }}
        ports: {{}}
        contract: {{ id: onnx-genai.token-policy, version: "1.0" }}
      arbitrary_drafter_name:
        implementation: {{ kind: onnx, artifact: proposer.onnx }}
        ports:
          inputs:
            fused_features:
              dtype: float32
              shape: [batch, sequence, 4]
              batch_layout: {{ kind: request_aligned, axis: 0 }}
            noisy_rows:
              dtype: float32
              shape: [batch, block, 2]
              batch_layout: {{ kind: request_aligned, axis: 0 }}
            holes:
              dtype: bool
              shape: [batch, block]
              batch_layout: {{ kind: request_aligned, axis: 0 }}
            absolute_positions:
              dtype: int64
              shape: [batch, total]
              batch_layout: {{ kind: request_aligned, axis: 0 }}
            visible:
              dtype: int64
              shape: [batch, total]
              batch_layout: {{ kind: request_aligned, axis: 0 }}
            shared_lm_head:
              dtype: float32
              shape: [2, 13]
              batch_layout: {{ kind: shared }}
            past_draft:
              dtype: float32
              shape: [batch, state_sequence, 2]
              batch_layout: {{ kind: request_aligned, axis: 0 }}
          outputs:
            selected_ids:
              dtype: int64
              shape: [batch, proposal]
              batch_layout: {{ kind: request_aligned, axis: 0 }}
{probability_port}
            present_draft:
              dtype: float32
              shape: [batch, state_sequence, 2]
              batch_layout: {{ kind: request_aligned, axis: 0 }}
      arbitrary_target_name:
        implementation: {{ kind: onnx, artifact: target.onnx }}
        ports:
          inputs:
            tokens:
              dtype: int64
              shape: [batch, verify]
              batch_layout: {{ kind: request_aligned, axis: 0 }}
            past_target:
              dtype: float32
              shape: [batch, state_sequence, 2]
              batch_layout: {{ kind: request_aligned, axis: 0 }}
          outputs:
            layer_alpha:
              dtype: float32
              shape: [batch, sequence, 2]
              batch_layout: {{ kind: request_aligned, axis: 0 }}
            layer_omega:
              dtype: float32
              shape: [batch, sequence, 2]
              batch_layout: {{ kind: request_aligned, axis: 0 }}
            exact_scores:
              dtype: float32
              shape: [batch, verify, 13]
              batch_layout: {{ kind: request_aligned, axis: 0 }}
            present_target:
              dtype: float32
              shape: [batch, state_sequence, 2]
              batch_layout: {{ kind: request_aligned, axis: 0 }}
          roles:
            layer_alpha: hidden_states
            layer_omega: hidden_states
            exact_scores: logits
    state:
      target_cache:
        contract: {{ dtype: float32, shape: [batch, state_sequence, 2], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        scope: invocation
        initializer: request.target_cache
        recurrence: {{ kind: invariant }}
        management: runtime
        release_boundary: invocation
        service_group: target_cache
      draft_cache:
        contract: {{ dtype: float32, shape: [batch, state_sequence, 2], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        scope: invocation
        initializer: request.draft_cache
        recurrence: {{ kind: invariant }}
        management: runtime
        release_boundary: invocation
        service_group: draft_cache
    steps:
      - kind: invoke
        component: arbitrary_target_name
        inputs: {{ tokens: request.target_tokens, past_target: request.target_cache }}
        outputs:
          layer_alpha: target.alpha
          layer_omega: target.omega
          exact_scores: target.scores
          present_target: target.present
      - kind: invoke
        component: arbitrary_drafter_name
        inputs:
          fused_features: request.target_hidden
          noisy_rows: request.noise
          holes: request.masked
          absolute_positions: request.positions
          visible: request.attention
          shared_lm_head: request.output_projection
          past_draft: request.draft_cache
        outputs:
          selected_ids: proposal.tokens
{probability_binding}          present_draft: proposal.present
      - {{ kind: emit, value: target.scores, output: target_logits, mode: replace }}
    serving:
      active: request.active
      done: request.done
      accepted_len: request.accepted_len
      state_service:
        groups:
          target_cache:
            kind: full_attention
            sequence_axis: 1
            layout: bsh
            update: {{ kind: append }}
            capabilities: {{ rollback_positions: 8, snapshot: true, fork: true, cascade: [draft_cache] }}
            ports:
              arbitrary_target_name:
                target_cache: {{ input: past_target, output: present_target }}
          draft_cache:
            kind: full_attention
            sequence_axis: 1
            layout: bsh
            update: {{ kind: append }}
            capabilities: {{ rollback_positions: 8, snapshot: true, fork: true, cascade: [target_cache] }}
            ports:
              arbitrary_drafter_name:
                draft_cache: {{ input: past_draft, output: present_draft }}
speculative:
  identity: onnx-genai.speculative
  version: "1"
  proposer: arbitrary_drafter_name
  target: arbitrary_target_name
  proposal_execution:
    kind: dflash_flat_block
    version: "{contract_version}"
    conditioning:
      sources:
        - {{ component: arbitrary_target_name, output: layer_alpha }}
        - {{ component: arbitrary_target_name, output: layer_omega }}
      proposer_input: fused_features
      combination: {{ kind: concatenate, axis: 2 }}
    block:
      noise_embeddings_input: noisy_rows
      masked_positions_input: holes
      position_ids_input: absolute_positions
      attention_mask_input: visible
      anchor_position: 0
      first_candidate_position: 1
      mask_token_id: 12
    outputs:
      candidate_tokens: selected_ids
{probability_output}      verifier_logits: {{ component: arbitrary_target_name, output: exact_scores }}
    shared_weights:
      input_embedding: {{ component: arbitrary_target_name, table: token_embedding }}
      output_projection:
        component: arbitrary_target_name
        initializer: lm_head
        proposer_input: shared_lm_head
        layout: hidden_vocabulary
    draft_private_state: [draft_cache]
    accepted_prefix_state:
      target_cache: {{ kind: sequence, source: {{ component: arbitrary_target_name, output: present_target }} }}
      draft_cache: {{ kind: sequence, source: {{ component: arbitrary_drafter_name, output: present_draft }} }}
    structure: {{ kind: base }}
  shared_weights:
    - {{ component: arbitrary_target_name, initializer: token_embedding }}
    - {{ component: arbitrary_target_name, initializer: lm_head }}
  vocabulary: {{ kind: identical }}
  max_proposal_width: 8
  distribution_preserving: true
  verification:
    target_output: {{ component: arbitrary_target_name, output: exact_scores }}
    accepted_path: {{ kind: runtime, binding: accepted_prefix }}
  rollback_state: [target_cache, draft_cache]
"#
    )
}

#[test]
fn dflash_uses_exact_version_schema_and_extension_registry() {
    let metadata = parse_metadata(&document("v1.6", "1", true), Some("yaml")).expect("parses");
    validate_metadata(&metadata).expect("valid DFlash contract");
    assert!(
        find(DFLASH_FLAT_BLOCK_V1.identity, DFLASH_FLAT_BLOCK_V1.version).is_some(),
        "DFlash v1 must be an exact registered semantic extension"
    );

    let old_reader_contract =
        parse_metadata(&document("v1.5", "1", true), Some("yaml")).expect("parses");
    let errors = validate_metadata(&old_reader_contract).expect_err("v1.5 cannot claim DFlash");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("schema version v1.6")),
        "{errors:#?}"
    );
}

#[test]
fn unknown_or_implicit_structural_versions_fail_closed() {
    let unknown = parse_metadata(&document("v1.6", "17", true), Some("yaml")).expect("parses");
    let errors = validate_metadata(&unknown).expect_err("unknown DFlash version");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("unsupported DFlash") && error.contains("'17'")),
        "{errors:#?}"
    );

    let implicit = document("v1.6", "1", true).replace(
        "structure: { kind: base }",
        "structure:\n      kind: selector_convolution_v1\n      selector:\n        selected_tokens_output: selected_ids\n        candidate_ids_output: selected_ids\n        top_k: 1\n        rank: 2\n      convolution:\n        kernel_size: 2\n        group_size: 1\n        first_position_reads_anchor: true",
    );
    let metadata = parse_metadata(&implicit, Some("yaml")).expect("parses");
    let errors = validate_metadata(&metadata).expect_err("v2 structure cannot hide under v1");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("require exact version '2'")),
        "{errors:#?}"
    );
}

#[test]
fn selector_and_convolution_semantics_require_and_admit_exact_version_two() {
    let document = document("v1.6", "2", false)
        .replace(
            "            selected_ids:\n              dtype: int64\n              shape: [batch, proposal]\n              batch_layout: { kind: request_aligned, axis: 0 }",
            "            selected_ids:\n              dtype: int64\n              shape: [batch, proposal]\n              batch_layout: { kind: request_aligned, axis: 0 }\n            selector_candidates:\n              dtype: int64\n              shape: [batch, proposal, 3]\n              batch_layout: { kind: request_aligned, axis: 0 }\n            selector_probabilities:\n              dtype: float32\n              shape: [batch, proposal, 3]\n              batch_layout: { kind: request_aligned, axis: 0 }",
        )
        .replace(
            "          selected_ids: proposal.tokens",
            "          selected_ids: proposal.tokens\n          selector_candidates: proposal.selector_candidates\n          selector_probabilities: proposal.selector_probabilities",
        )
        .replace(
            "    structure: { kind: base }",
            "    structure:\n      kind: selector_convolution_v1\n      selector:\n        selected_tokens_output: selected_ids\n        candidate_ids_output: selector_candidates\n        conditional_probabilities_output: selector_probabilities\n        top_k: 3\n        rank: 2\n      convolution:\n        kernel_size: 2\n        group_size: 1\n        first_position_reads_anchor: true",
        );
    let metadata = parse_metadata(&document, Some("yaml")).expect("version-2 document parses");
    validate_metadata(&metadata).expect("explicit selector/convolution version is admitted");
}

#[test]
fn conditioning_and_state_provenance_are_not_inferred_from_names() {
    let forged = document("v1.6", "1", true).replace(
        "layer_alpha: hidden_states",
        "layer_alpha: encoder_hidden_states",
    );
    let metadata = parse_metadata(&forged, Some("yaml")).expect("parses");
    let errors = validate_metadata(&metadata).expect_err("wrong semantic provenance");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("lacks the hidden_states output role")),
        "{errors:#?}"
    );

    let omitted = document("v1.6", "1", true).replace(
        "rollback_state: [target_cache, draft_cache]",
        "rollback_state: [target_cache]",
    );
    let metadata = parse_metadata(&omitted, Some("yaml")).expect("parses");
    let errors = validate_metadata(&metadata).expect_err("draft state cannot escape rollback");
    assert!(
        errors.iter().any(|error| {
            error.contains("accepted_prefix_state")
                || (error.contains("draft_cache") && error.contains("rollback"))
        }),
        "{errors:#?}"
    );
}

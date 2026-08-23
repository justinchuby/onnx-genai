//! A declared conversation, checked at the document.
//!
//! `scope: session` says a value outlives its invocation. It does not say how
//! the next invocation reaches it, and a package that leaves that unanswered
//! advertises continuity it silently does not have — which is how a multi-turn
//! conversation degrades into a model that forgets what it was told. These pin
//! the checks that refuse such a document instead of the third turn.

use onnx_genai_metadata::{InferenceMetadata, validate_metadata};

fn document(state: &str, manifest: &str) -> String {
    format!(
        r#"
schema_version: v1
pipeline:
  workflow:
    manifest:
      capabilities: [workflow_ssa, typed_emit, {manifest}]
    inputs:
      request.input_ids:
        contract: {{ dtype: int64, rank: 2, shape: [batch, sequence], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: runtime, version: '1.0', role: prompt_tokens }}
        source: {{ kind: request }}
        required: true
      package.max_context:
        contract: {{ dtype: int64, rank: 1, shape: [1] }}
        role: {{ kind: opaque }}
        source: {{ kind: literal }}
        required: false
        default: 64
    outputs:
      tokens:
        contract: {{ dtype: int64, rank: 2, shape: [batch, generated], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: tokens
        stage: pre_adapter
    components:
      decoder:
        implementation: {{ kind: onnx, artifact: model.onnx }}
        ports:
          inputs:
            input_ids: {{ dtype: int64, rank: 2, shape: [batch, sequence], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
          outputs:
            next_tokens: {{ dtype: int64, rank: 2, shape: [batch, generated], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
          roles:
            input_ids: token_ids
    state:
{state}
    steps:
      - kind: invoke
        component: decoder
        inputs: {{ input_ids: request.input_ids }}
        outputs: {{ next_tokens: decoder.tokens }}
      - kind: emit
        value: decoder.tokens
        output: tokens
        mode: replace
"#
    )
}

fn conversation_cell(overrides: &[(&str, &str)]) -> String {
    let mut fields = vec![
        ("class", "semantic".to_string()),
        ("scope", "session".to_string()),
        ("initializer", "request.input_ids".to_string()),
        (
            "recurrence",
            "{ kind: bounded, axis: 1, max: package.max_context }".to_string(),
        ),
        ("management", "runtime".to_string()),
        ("release_boundary", "session".to_string()),
        (
            "session",
            "{ policy: exclusive, continuation: { kind: prompt_prefix, prompt_input: \
             request.input_ids, tokens_output: tokens } }"
                .to_string(),
        ),
    ];
    for (key, value) in overrides {
        match fields.iter_mut().find(|(name, _)| name == key) {
            Some(field) => field.1 = (*value).to_string(),
            None => fields.push((key, (*value).to_string())),
        }
    }
    let body = fields
        .iter()
        .map(|(key, value)| format!("        {key}: {value}"))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "      conversation:\n        contract: {{ dtype: int64, rank: 2, shape: [batch, \
         conversation_length], batch_layout: {{ kind: request_aligned, axis: 0 }} }}\n{body}\n"
    )
}

fn errors(document: &str) -> Vec<String> {
    let metadata: InferenceMetadata =
        serde_yaml::from_str(document).expect("workflow metadata parses");
    match validate_metadata(&metadata) {
        Ok(()) => Vec::new(),
        Err(errors) => errors,
    }
}

fn rejects(document: &str, fragment: &str) {
    let reported = errors(document);
    assert!(
        reported.iter().any(|error| error.contains(fragment)),
        "expected an error containing {fragment:?}, got {reported:?}"
    );
}

#[test]
fn a_well_formed_conversation_validates() {
    let reported = errors(&document(
        &conversation_cell(&[]),
        "session_state_lease, bounded_state_recurrence",
    ));
    assert!(reported.is_empty(), "{reported:?}");
}

/// The lease and the capability are one statement: a reader that cannot honour
/// leased state must be able to see that it is being asked to.
#[test]
fn a_conversation_requires_the_session_lease_capability() {
    rejects(
        &document(&conversation_cell(&[]), "typed_emit"),
        "session_state_lease",
    );
}

#[test]
fn a_continuation_must_be_session_scoped() {
    rejects(
        &document(
            &conversation_cell(&[("scope", "invocation"), ("release_boundary", "invocation")]),
            "session_state_lease",
        ),
        "not session-scoped",
    );
}

/// A conversation released with its invocation is not a conversation.
#[test]
fn a_continuation_must_survive_its_invocation() {
    rejects(
        &document(
            &conversation_cell(&[("release_boundary", "invocation")]),
            "session_state_lease",
        ),
        "release_boundary: session",
    );
}

/// Advisory state may be dropped; a conversation may not.
#[test]
fn a_continuation_must_be_semantic() {
    rejects(
        &document(
            &conversation_cell(&[("class", "advisory")]),
            "session_state_lease, advisory_state",
        ),
        "class: semantic",
    );
}

#[test]
fn a_continuation_must_grow() {
    rejects(
        &document(
            &conversation_cell(&[("recurrence", "{ kind: invariant }")]),
            "session_state_lease",
        ),
        "a conversation grows with every turn",
    );
}

/// The prompt input a continuation prefixes has to be the prompt.
#[test]
fn a_continuation_must_name_the_prompt_tokens_input() {
    rejects(
        &document(
            &conversation_cell(&[(
                "session",
                "{ policy: exclusive, continuation: { kind: prompt_prefix, prompt_input: \
                 package.max_context, tokens_output: tokens } }",
            )]),
            "session_state_lease",
        ),
        "does not carry the prompt_tokens runtime role",
    );
    rejects(
        &document(
            &conversation_cell(&[(
                "session",
                "{ policy: exclusive, continuation: { kind: prompt_prefix, prompt_input: \
                 request.nothing, tokens_output: tokens } }",
            )]),
            "session_state_lease",
        ),
        "not a declared workflow input",
    );
}

#[test]
fn a_continuation_must_name_the_tokens_output_it_accumulates() {
    rejects(
        &document(
            &conversation_cell(&[(
                "session",
                "{ policy: exclusive, continuation: { kind: prompt_prefix, prompt_input: \
                 request.input_ids, tokens_output: nothing } }",
            )]),
            "session_state_lease",
        ),
        "not a declared workflow output",
    );
}

/// A package has one conversation.
#[test]
fn two_continuations_leave_no_answer_about_which_one_a_turn_continues() {
    let two = format!(
        "{}{}",
        conversation_cell(&[]),
        conversation_cell(&[]).replace("conversation:", "second_conversation:")
    );
    rejects(
        &document(&two, "session_state_lease, bounded_state_recurrence"),
        "one conversation",
    );
}

/// A leased cell that binds a state service group must reach one.
///
/// This is the fail-closed half of the group declaration: a package claiming
/// its session-scoped cache is carried by group `decoder_cache` when no such
/// group exists, or when no alias in it names the cell, is claiming continuity
/// that has nothing behind it.
#[test]
fn a_session_cell_binding_an_unknown_state_group_is_refused() {
    let cell = conversation_cell(&[("service_group", "decoder_cache")]);
    rejects(
        &document(&cell, "session_state_lease, bounded_state_recurrence"),
        "which pipeline.workflow.serving.state_service.groups does not declare",
    );
}

/// A bound that names nothing is not a bound.
///
/// The runtime reads it before the turn that would exceed it runs and again when
/// the turn completes, and it is the only check on a continuation's length — a
/// continuation is not loop-carried, so it never reaches the carry path where a
/// recurrence value is otherwise resolved. A symbol no input declares would make
/// the check silently pass.
#[test]
fn a_continuations_bound_must_name_a_declared_input() {
    rejects(
        &document(
            &conversation_cell(&[(
                "recurrence",
                "{ kind: bounded, axis: 1, max: package.context_window }",
            )]),
            "session_state_lease, bounded_state_recurrence",
        ),
        "not a declared workflow input",
    );
}

/// A bound a request may omit is not a bound either.
#[test]
fn a_continuations_bound_must_have_a_value_by_the_time_a_turn_is_admitted() {
    let no_default = document(
        &conversation_cell(&[]),
        "session_state_lease, bounded_state_recurrence",
    )
    .replace(
        "        required: false\n        default: 64\n",
        "        required: false\n",
    );
    rejects(&no_default, "declares no default");
}

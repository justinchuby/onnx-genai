//! The canonical autoregressive decode loop.
//!
//! A bare-decoder package has no serialized workflow; the loader compiles its
//! `model.io` into one (`onnx_genai_metadata::canonical`). This module executes
//! that lowered workflow, so a plain decoder and an authored workflow reach the
//! same runtime instead of two.
//!
//! # What runs the workflow, and what runs the model
//!
//! [`run_canonical_decode`] reads its loop body *out of the workflow*
//! ([`resolve_body`]) and dispatches each step by the component's declared
//! contract id. The spec is therefore executed, not merely attested: a body that
//! names a contract this runtime does not implement, or that never applies a
//! policy, fails here rather than silently doing something else.
//!
//! The loop is the runtime's — iteration bound, context-limit stop, stop
//! decision, per-request decode state. The executor behind
//! `onnx-genai.autoregressive-decode` owns exactly one thing: a decoder forward
//! pass, and with it the paged / share-buffer / CUDA-graph KV. That boundary is
//! deliberate: the lowered workflow declares its KV `management: runtime`, so
//! those buffers never round-trip through the interpreter as SSA values and
//! device residency is untouched.
//!
//! Sampling, stopping, logprobs, and commit are not reimplemented here. They are
//! [`select_and_commit_step`], the same function `step_decode_loop` calls.
//!
//! # Scope: which loops this replaced, and which remain
//!
//! `run_decode_loop` — the second single-row token loop that only the direct
//! `Engine` path could drive — is deleted. Every *single-row autoregressive*
//! request now runs here: the ORT session path, both native decode sessions, and
//! the workflow path.
//!
//! Two loops remain, and they are different algorithms rather than duplicate
//! implementations of this one:
//!
//! * **continuous batching** (`batched.rs`) advances N rows per forward pass, so
//!   it cannot be a single-row loop by construction;
//! * **speculative decoding** (`speculative/mod.rs`, `native_speculative.rs`)
//!   proposes a block and verifies it, so its iteration is a block rather than a
//!   token.
//!
//! Both drive the *same* policy primitives this loop does —
//! `processors::select_next_token*`, `logprob_for_token`,
//! `commit_selected_token`, `ensure_constrained_finish`, `finish_result` — so
//! there is one sampling/stopping implementation across all three. What differs
//! is the shape of the iteration, which is inherent to what they do.

use anyhow::Context as _;
use onnx_genai_metadata::WorkflowStep;

use crate::config::{FinishReason, GenerateOptions, GenerateResult, GenerateTokenCallback};
use crate::decode_loop::{
    DecodeLoopBackend, DecodeLoopState, finish_result, forward_step, reached_context_limit,
    select_and_commit_step,
};
use crate::logits::ProcessorChain;
use crate::processors::ensure_constrained_finish;
use onnx_genai_ort::Tokenizer;

/// Everything one canonical decode loop needs beyond its executor.
pub(crate) struct CanonicalDecodeRequest<'a> {
    pub(crate) options: &'a GenerateOptions,
    pub(crate) chain: &'a ProcessorChain,
    pub(crate) tokenizer: &'a Tokenizer,
    pub(crate) max_context: Option<usize>,
    /// The canonical workflow this request executes.
    ///
    /// Not decoration: [`run_canonical_decode`] reads the loop body out of it
    /// and dispatches each step by the component's declared contract id, so the
    /// lowered spec determines what runs and in what order. A body this runtime
    /// cannot implement is an error, not a silent fallback.
    pub(crate) workflow: &'a onnx_genai_metadata::WorkflowSpec,
}

/// One step of the canonical loop body, resolved from the workflow.
enum BodyStep {
    /// `onnx-genai.autoregressive-decode`: one decoder forward pass.
    Decode,
    /// `onnx-genai.token-policy`: score, commit, stop-check.
    TokenPolicy,
    /// Publish the step's token into a declared package output.
    Emit,
}

/// Resolve the canonical loop's body from the workflow.
///
/// Reading the body from the spec is what makes the lowered workflow the thing
/// that *runs* rather than a load-time attestation: declare a component whose
/// contract this runtime does not implement, or a body that never applies a
/// policy, and execution fails here instead of quietly doing something else.
fn resolve_body(workflow: &onnx_genai_metadata::WorkflowSpec) -> anyhow::Result<Vec<BodyStep>> {
    let loop_body = workflow
        .steps
        .iter()
        .find_map(|step| match step {
            WorkflowStep::Loop { steps, .. } => Some(steps),
            _ => None,
        })
        .context(
            "the canonical workflow declares no loop; an autoregressive decoder's token stream \
             is a loop by construction",
        )?;
    let mut body = Vec::with_capacity(loop_body.len());
    for step in loop_body {
        match step {
            WorkflowStep::Invoke { component, .. } => {
                let declaration = workflow.components.get(component).with_context(|| {
                    format!("canonical loop invokes undeclared component '{component}'")
                })?;
                let contract = declaration.contract.as_ref().with_context(|| {
                    format!(
                        "canonical component '{component}' declares no contract, so this runtime \
                         cannot tell what it is asked to execute"
                    )
                })?;
                match contract.id.as_str() {
                    onnx_genai_metadata::AUTOREGRESSIVE_DECODE_CONTRACT => {
                        body.push(BodyStep::Decode)
                    }
                    onnx_genai_metadata::TOKEN_POLICY_CONTRACT => body.push(BodyStep::TokenPolicy),
                    other => anyhow::bail!(
                        "canonical loop invokes component '{component}' with contract '{other}', \
                         which this runtime does not implement"
                    ),
                }
            }
            WorkflowStep::Emit { .. } => body.push(BodyStep::Emit),
            other => anyhow::bail!(
                "the canonical loop body may only invoke components and emit; found {other:?}"
            ),
        }
    }
    anyhow::ensure!(
        body.iter().any(|step| matches!(step, BodyStep::Decode)),
        "the canonical loop body runs no decoder forward pass"
    );
    anyhow::ensure!(
        body.iter()
            .any(|step| matches!(step, BodyStep::TokenPolicy)),
        "the canonical loop body applies no token policy, so nothing would select or stop"
    );
    Ok(body)
}

/// Drive the canonical `Loop { decode; token_policy; emit }` to completion.
///
/// The loop is the runtime's: iteration bound, the context-limit stop, the stop
/// decision, and the per-request decode state live here. The body comes from the
/// lowered workflow and each step is dispatched by its declared contract id.
pub(crate) fn run_canonical_decode<B: DecodeLoopBackend + ?Sized>(
    backend: &mut B,
    state: &mut DecodeLoopState,
    request: CanonicalDecodeRequest<'_>,
    mut callback: Option<&mut GenerateTokenCallback<'_>>,
) -> anyhow::Result<GenerateResult> {
    let CanonicalDecodeRequest {
        options,
        chain,
        tokenizer,
        max_context,
        workflow,
    } = request;
    let body = resolve_body(workflow)?;
    state.generated_tokens.reserve(
        options
            .max_new_tokens
            .saturating_sub(state.generated_tokens.len()),
    );
    if let Some(logprobs) = state.logprobs.as_mut() {
        logprobs.reserve(options.max_new_tokens.saturating_sub(logprobs.len()));
    }

    while state.generated_tokens.len() < options.max_new_tokens {
        let _step_span = if state.generated_tokens.is_empty() {
            onnx_genai_ort::prof_span!("loop.prefill")
        } else {
            onnx_genai_ort::prof_span!("loop.step")
        };
        // The context bound is a loop-level stop, checked before the body runs:
        // a step that cannot fit is not attempted.
        if reached_context_limit(backend.context_len(), max_context) {
            ensure_constrained_finish(options, &state.generated_text, FinishReason::Length)?;
            return finish_result(
                tokenizer,
                &state.generated_tokens,
                FinishReason::Length,
                state.prefix_cache_hit_len,
                state.logprobs.as_deref(),
            );
        }

        let mut forward = None;
        let mut finish = None;
        for step in &body {
            match step {
                BodyStep::Decode => {
                    forward = Some(forward_step(backend, state, options, chain)?);
                }
                BodyStep::TokenPolicy => {
                    let forward = forward.take().context(
                        "the canonical loop applied its token policy before any decoder forward \
                         pass produced logits",
                    )?;
                    let (_token, reason) = select_and_commit_step(
                        backend,
                        state,
                        options,
                        chain,
                        tokenizer,
                        forward,
                        callback.as_deref_mut(),
                    )?;
                    finish = reason;
                }
                // The committed token stream is published by `finish_result`
                // from the policy's own state, so the emit step names where it
                // lands rather than moving bytes here.
                BodyStep::Emit => {}
            }
        }

        // `continue_when`: the policy reported no finish reason.
        if let Some(finish_reason) = finish {
            return finish_result(
                tokenizer,
                &state.generated_tokens,
                finish_reason,
                state.prefix_cache_hit_len,
                state.logprobs.as_deref(),
            );
        }
    }

    // `max_iterations` exhausted.
    ensure_constrained_finish(options, &state.generated_text, FinishReason::MaxTokens)?;
    finish_result(
        tokenizer,
        &state.generated_tokens,
        FinishReason::MaxTokens,
        state.prefix_cache_hit_len,
        state.logprobs.as_deref(),
    )
}

/// Assert the package this runtime holds really is executable as the canonical
/// workflow before anything runs.
///
/// A lowered decoder must have produced a workflow whose two component contracts
/// are the ones this loop implements. Checking it here means a package that
/// somehow reached generation without a canonical form fails with the contract
/// it was missing, rather than silently taking a path that no longer exists.
pub(crate) fn assert_canonical_contracts(
    workflow: &onnx_genai_metadata::WorkflowSpec,
) -> anyhow::Result<()> {
    for (component, expected) in [
        (
            onnx_genai_metadata::DECODER_COMPONENT,
            onnx_genai_metadata::AUTOREGRESSIVE_DECODE_CONTRACT,
        ),
        (
            onnx_genai_metadata::POLICY_COMPONENT,
            onnx_genai_metadata::TOKEN_POLICY_CONTRACT,
        ),
    ] {
        let declared = workflow
            .components
            .get(component)
            .with_context(|| format!("the canonical workflow declares no '{component}' component"))?
            .contract
            .as_ref()
            .with_context(|| format!("canonical component '{component}' declares no contract"))?;
        anyhow::ensure!(
            declared.id == expected,
            "canonical component '{component}' declares contract '{}', but this runtime \
             implements '{expected}'",
            declared.id
        );
    }
    Ok(())
}

/// The canonical workflow a test double stands in for.
///
/// A hand-built `Engine` skips the loader, so it would otherwise reach
/// generation with no canonical workflow and trip the guard that exists to
/// catch exactly that. Giving it the same lowering a real minimal decoder gets
/// keeps the double honest instead of weakening the guard for everyone.
#[cfg(test)]
pub(crate) fn test_canonical_workflow() -> onnx_genai_metadata::WorkflowSpec {
    let io: onnx_genai_metadata::ModelIoSpec =
        serde_yaml::from_str("token_input: input_ids\nlogits_output: logits\n")
            .expect("minimal decoder ABI");
    onnx_genai_metadata::lower_decoder_abi(&io).expect("minimal decoder lowers")
}

//! The interpreter's canonical autoregressive decode loop.
//!
//! A bare-decoder package has no serialized workflow; the loader compiles its
//! `model.io` into one (`onnx_genai_metadata::canonical`). This module is what
//! *executes* that lowered workflow, so a plain decoder and an authored workflow
//! reach the same runtime instead of two.
//!
//! # What the interpreter owns, and what the executor owns
//!
//! The loop is the interpreter's: iteration count, the stop decision, the
//! emitted token stream, and the per-request decode state all live here. The
//! executor owns exactly one thing — a decoder forward pass — behind
//! [`DecodeLoopBackend`], which is also where the paged / share-buffer /
//! CUDA-graph KV lives. That boundary is the whole point: the lowered workflow
//! declares its KV `management: runtime`, so those buffers never round-trip
//! through the interpreter as SSA values and device residency is untouched.
//!
//! Sampling, stopping, logprobs, and commit are *not* reimplemented here. They
//! are [`select_and_commit_step`], the same function `step_decode_loop` calls,
//! so there is one policy no matter which loop drives it.
//!
//! # Why this is not `run_decode_loop`
//!
//! `run_decode_loop` was a second token loop that only the direct `Engine` path
//! could drive. It is gone. What remains is one loop, here, reached by every
//! generated request.

use anyhow::Context as _;

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
}

/// Drive the canonical `Loop { decode; token_policy; emit }` to completion.
///
/// Structurally this is the lowered workflow's loop: `max_iterations` is
/// `max_new_tokens`, the body is one decoder invocation followed by the token
/// policy, and `continue_when` is "the policy reported no finish reason". It is
/// written as a Rust loop over those two steps rather than as a generic SSA walk
/// because the decoder's state is runtime-managed by declaration — there are no
/// SSA values to thread between iterations, so a generic walk would have nothing
/// to carry and would only obscure where the token stream comes from.
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
    } = request;
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

        // Body step 1 — `onnx-genai.autoregressive-decode`: one forward pass.
        let forward = forward_step(backend, state, options, chain)?;
        // Body step 2 — `onnx-genai.token-policy`: score, commit, stop-check.
        let (_token, finish) = select_and_commit_step(
            backend,
            state,
            options,
            chain,
            tokenizer,
            forward,
            callback.as_deref_mut(),
        )?;
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

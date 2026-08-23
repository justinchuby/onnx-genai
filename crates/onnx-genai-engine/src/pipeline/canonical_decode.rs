//! The canonical autoregressive decode loop.
//!
//! A bare-decoder package has no serialized workflow; the loader compiles its
//! `model.io` into one (`onnx_genai_metadata::canonical`). This module executes
//! that lowered workflow, so a plain decoder is no longer a *second kind of
//! package* that decodes without one.
//!
//! To be precise about what that does and does not mean: a lowered decoder and
//! an authored workflow both execute a `WorkflowSpec`, but not through the same
//! executor. An authored workflow runs the general interpreter
//! (`pipeline::workflow::run_workflow_node`) and may express token selection as
//! an in-graph ONNX policy; a lowered decoder runs *this* body so it keeps the
//! rich Rust sampler, paged KV, sessions, and speculative decode — none of which
//! has an in-graph representation. What is unified is that there is exactly one
//! single-row autoregressive loop and one next-token policy, not that there is
//! one executor for every package shape.
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
//! [`select_and_commit_step`], which every loop in the crate calls.
//!
//! # Scope: which loops this replaced, and which remain
//!
//! `run_decode_loop` and `step_decode_loop` — the second single-row token loop
//! and its per-step twin, which only the direct `Engine` path could drive — are
//! both deleted. Every *single-row autoregressive* request now runs this body:
//! the ORT session path, both native decode sessions, the scheduler's
//! prioritized drive (through [`step_canonical_body`]), and the workflow path.
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
#[derive(Debug)]
enum BodyStep {
    /// `onnx-genai.autoregressive-decode`: one decoder forward pass.
    Decode,
    /// `onnx-genai.token-policy`: score, commit, stop-check.
    TokenPolicy,
    /// Publish the step's token into a declared package output.
    ///
    /// Carries the output it targets so the step is checked rather than
    /// assumed: an emit naming an undeclared output is a broken spec, and a
    /// loop that emits nothing produces no token stream for a caller to read.
    Emit { output: String },
}

/// Resolve the canonical loop's body from the workflow.
///
/// Reading the body from the spec is what makes the lowered workflow the thing
/// that *runs* rather than a load-time attestation: declare a component whose
/// contract this runtime does not implement, or a body that never applies a
/// policy, and execution fails here instead of quietly doing something else.
fn resolve_body(workflow: &onnx_genai_metadata::WorkflowSpec) -> anyhow::Result<Vec<BodyStep>> {
    // The decoder is recognized structurally — the sole component consuming the
    // autoregressive sequence and producing logits — so no component name, model
    // name, or bespoke contract decides what runs. A package renames its
    // component freely and this still works.
    let decoder = onnx_genai_metadata::sole_decoder_component(workflow).context(
        "this workflow presents no single decoder component, so it is not a single-row \
         autoregressive package and must be executed by the generic interpreter",
    )?;
    let loop_body = workflow
        .steps
        .iter()
        .find_map(|step| match step {
            WorkflowStep::Loop { steps, .. } => Some(steps),
            _ => None,
        })
        .context(
            "the workflow declares no loop; an autoregressive decoder's token stream \
             is a loop by construction",
        )?;
    let mut body = Vec::with_capacity(loop_body.len());
    for step in loop_body {
        match step {
            WorkflowStep::Invoke { component, .. } if component == decoder => {
                body.push(BodyStep::Decode)
            }
            WorkflowStep::Invoke { component, .. } => {
                let declaration = workflow.components.get(component).with_context(|| {
                    format!("the loop invokes undeclared component '{component}'")
                })?;
                let contract = declaration.contract.as_ref().with_context(|| {
                    format!(
                        "component '{component}' is neither the decoder nor declares a contract, \
                         so this runtime cannot tell what it is asked to execute"
                    )
                })?;
                match contract.id.as_str() {
                    onnx_genai_metadata::decoder_workflow::TOKEN_POLICY_CONTRACT => {
                        body.push(BodyStep::TokenPolicy)
                    }
                    other => anyhow::bail!(
                        "the loop invokes component '{component}' with contract '{other}', \
                         which this runtime does not implement"
                    ),
                }
            }
            WorkflowStep::Emit { output, .. } => {
                anyhow::ensure!(
                    workflow.outputs.contains_key(output),
                    "the loop emits into '{output}', which the workflow does not \
                     declare as an output"
                );
                body.push(BodyStep::Emit {
                    output: output.clone(),
                })
            }
            other => anyhow::bail!(
                "the generation loop body may only invoke components and emit; found {other:?}"
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
    anyhow::ensure!(
        body.iter().any(|step| matches!(
            step,
            BodyStep::Emit { output }
                if output == onnx_genai_metadata::decoder_workflow::TOKENS_OUTPUT
        )),
        "the canonical loop body never emits '{}', so it would decode without producing the \
         token stream the caller reads",
        onnx_genai_metadata::decoder_workflow::TOKENS_OUTPUT
    );
    Ok(body)
}

/// The canonical loop body, resolved once per request.
///
/// Held by a caller that advances a request a step at a time (the scheduler's
/// prioritized drive) so it resolves the workflow once rather than per token,
/// and by [`run_canonical_decode`] for its own loop.
pub(crate) struct CanonicalBody {
    steps: Vec<BodyStep>,
}

impl CanonicalBody {
    /// Resolve the loop body from a canonical workflow.
    pub(crate) fn resolve(workflow: &onnx_genai_metadata::WorkflowSpec) -> anyhow::Result<Self> {
        Ok(Self {
            steps: resolve_body(workflow)?,
        })
    }
}

/// Run one iteration of the canonical loop body.
///
/// Returns the finish reason when the policy reported one. The context-limit
/// stop belongs to the loop, not the body, so a caller that owns the iteration
/// checks it before calling in — [`run_canonical_decode`] does, and so does the
/// prioritized drive.
pub(crate) fn step_canonical_body<B: DecodeLoopBackend + ?Sized>(
    backend: &mut B,
    state: &mut DecodeLoopState,
    body: &CanonicalBody,
    options: &GenerateOptions,
    chain: &ProcessorChain,
    tokenizer: &Tokenizer,
    mut callback: Option<&mut GenerateTokenCallback<'_>>,
) -> anyhow::Result<Option<FinishReason>> {
    let mut forward = None;
    let mut finish = None;
    for step in &body.steps {
        match step {
            BodyStep::Decode => {
                forward = Some(forward_step(backend, state, options, chain)?);
            }
            BodyStep::TokenPolicy => {
                let forward = forward.take().context(
                    "the canonical loop applied its token policy before any decoder forward pass \
                     produced logits",
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
            // The committed token stream is published by `finish_result` from
            // the policy's own state, so the emit step names where it lands
            // rather than moving bytes here.
            // The emitted token stream is the `Vec<u32>` this loop returns and
            // the callback it already fired; `resolve_body` has checked the
            // step names the declared tokens output, so there is nothing
            // further to route here.
            BodyStep::Emit { .. } => {}
        }
    }
    Ok(finish)
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
    let body = CanonicalBody::resolve(workflow)?;
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
        let finish = step_canonical_body(
            backend,
            state,
            &body,
            options,
            chain,
            tokenizer,
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
    // Resolving the body *is* the check: it recognizes the decoder structurally,
    // requires a token policy this runtime implements, and requires the loop to
    // emit the token stream a caller reads. Anything a separate assertion could
    // add here would be a second, weaker statement of the same rule that could
    // drift from the one execution actually uses.
    resolve_body(workflow).map(|_| ())
}

/// The canonical workflow a test double stands in for.
///
/// A hand-built `Engine` skips the loader, so it would otherwise reach
/// generation with no canonical workflow and trip the guard that exists to
/// catch exactly that. Giving it the same lowering a real minimal decoder gets
/// keeps the double honest instead of weakening the guard for everyone.
#[cfg(test)]
pub(crate) fn test_canonical_workflow() -> onnx_genai_metadata::WorkflowSpec {
    let abi = onnx_genai_metadata::DecoderAbi {
        token_input: Some("input_ids".to_string()),
        logits_output: Some("logits".to_string()),
        kv_inputs: Some(vec!["past.key".to_string(), "past.value".to_string()]),
        kv_outputs: Some(vec!["present.key".to_string(), "present.value".to_string()]),
        ..onnx_genai_metadata::DecoderAbi::default()
    };
    onnx_genai_metadata::decoder_workflow::decoder_workflow(
        &abi,
        "model.onnx",
        &onnx_genai_metadata::decoder_workflow::DecoderFacts::default(),
    )
    .expect("a minimal decoder is expressible as a workflow")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Resolving a body is a check, not a formality.
    ///
    /// Each case mutates the real lowering into a spec that would still *load*
    /// but describes something this runtime cannot honestly execute, and
    /// asserts resolution refuses it. Without these, `resolve_body` could
    /// degrade into "read the spec, ignore what it says" and no test would
    /// notice.
    fn refuses(mutate: impl FnOnce(&mut onnx_genai_metadata::WorkflowSpec), expected: &str) {
        let mut workflow = test_canonical_workflow();
        mutate(&mut workflow);
        let error = resolve_body(&workflow)
            .expect_err("a body this runtime cannot execute must not resolve");
        let message = format!("{error:#}");
        assert!(
            message.contains(expected),
            "expected the refusal to explain '{expected}', got: {message}"
        );
    }

    fn loop_body(workflow: &mut onnx_genai_metadata::WorkflowSpec) -> &mut Vec<WorkflowStep> {
        workflow
            .steps
            .iter_mut()
            .find_map(|step| match step {
                WorkflowStep::Loop { steps, .. } => Some(steps),
                _ => None,
            })
            .expect("the canonical lowering declares a loop")
    }

    #[test]
    fn the_canonical_lowering_resolves() {
        let workflow = test_canonical_workflow();
        let body = resolve_body(&workflow).expect("the real lowering must resolve");
        assert!(body.iter().any(|step| matches!(step, BodyStep::Decode)));
        assert!(
            body.iter()
                .any(|step| matches!(step, BodyStep::TokenPolicy))
        );
        assert!(
            body.iter()
                .any(|step| matches!(step, BodyStep::Emit { .. }))
        );
    }

    #[test]
    fn a_body_without_a_decoder_is_refused() {
        refuses(
            |workflow| {
                loop_body(workflow).retain(|step| {
                    !matches!(
                        step,
                        WorkflowStep::Invoke { component, .. }
                            if component == onnx_genai_metadata::decoder_workflow::DECODER_COMPONENT
                    )
                });
            },
            "runs no decoder forward pass",
        );
    }

    #[test]
    fn a_body_without_a_token_policy_is_refused() {
        refuses(
            |workflow| {
                loop_body(workflow).retain(|step| {
                    !matches!(
                        step,
                        WorkflowStep::Invoke { component, .. }
                            if component == onnx_genai_metadata::decoder_workflow::POLICY_COMPONENT
                    )
                });
            },
            "applies no token policy",
        );
    }

    #[test]
    fn a_body_that_emits_nothing_is_refused() {
        refuses(
            |workflow| {
                loop_body(workflow).retain(|step| !matches!(step, WorkflowStep::Emit { .. }));
            },
            "never emits",
        );
    }

    #[test]
    fn an_emit_into_an_undeclared_output_is_refused() {
        refuses(
            |workflow| {
                for step in loop_body(workflow) {
                    if let WorkflowStep::Emit { output, .. } = step {
                        *output = "not_a_declared_output".to_string();
                    }
                }
            },
            "which the workflow does not declare as an output",
        );
    }

    /// The decoder is recognized structurally, but every *other* component in
    /// the loop must name a contract this runtime implements.
    #[test]
    fn an_unimplemented_contract_is_refused() {
        refuses(
            |workflow| {
                if let Some(component) = workflow
                    .components
                    .get_mut(onnx_genai_metadata::decoder_workflow::POLICY_COMPONENT)
                    && let Some(contract) = component.contract.as_mut()
                {
                    contract.id = "vendor.some-other-thing".to_string();
                }
            },
            "which this runtime does not implement",
        );
    }

    #[test]
    fn a_workflow_without_a_loop_is_refused() {
        refuses(
            |workflow| {
                workflow
                    .steps
                    .retain(|step| !matches!(step, WorkflowStep::Loop { .. }));
            },
            "declares no loop",
        );
    }
}

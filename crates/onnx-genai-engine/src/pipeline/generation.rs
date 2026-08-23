//! Autoregressive generation, executed by the workflow interpreter.
//!
//! A package that generates text declares a loop: a decoder step, a token
//! policy step, and an emit that publishes the selected token into the
//! package's `tokens` output. This module supplies the *executors* for the two
//! runtime-implemented steps and hands them to the interpreter. It does not own
//! a loop.
//!
//! That distinction is the whole point. The iteration bound, the liveness
//! predicate, the carried state, the emit gate and the stop are read from the
//! workflow by [`crate::pipeline::workflow`]'s `Loop` node, exactly as they are
//! for a composite pipeline. What a single-decoder package gets that a
//! composite one does not is a *faster executor for one declared step*: the
//! fused decode session, which owns paged KV, the device sampling fast paths
//! and the CUDA graph. It is selected because the component declares
//! `onnx-genai.autoregressive-decode` and this runtime registered an executor
//! for that contract — not because Rust inspected the package's shape.
//!
//! # What the executors own, and what they must not
//!
//! [`GenerationNodeHost`] advances exactly one node per call. Between the
//! decode node and the policy node it holds one thing: the forward pass's
//! outcome, which is that iteration's logits (or the token a device fast path
//! already selected). Everything else — how many iterations may run, whether
//! another one may, which value the emit publishes — belongs to the
//! interpreter.
//!
//! The decode node's `logits` output is deliberately *not* published into the
//! SSA environment. A fused session keeps them on device, and materializing
//! them on the host to satisfy a name nothing reads would undo the fast path it
//! exists for. [`validate_generation_workflow`] refuses at load if any step
//! other than a runtime-implemented one reads that value, so the omission is a
//! checked property of the package rather than an assumption about it.

use std::collections::{BTreeMap, HashSet};

use anyhow::Context as _;
use onnx_genai_metadata::decoder_workflow::{
    AUTOREGRESSIVE_DECODE_CONTRACT, TOKEN_POLICY_CONTRACT,
};
use onnx_genai_metadata::{WorkflowSpec, WorkflowStep};
use onnx_genai_ort::{DataType, Tokenizer, Value};

use crate::config::{FinishReason, GenerateOptions, GenerateResult, GenerateTokenCallback};
use crate::decode_loop::{
    DecodeLoopBackend, DecodeLoopState, ForwardOutcome, finish_result, forward_step,
    reached_context_limit, select_and_commit_step,
};
use crate::logits::ProcessorChain;
use crate::processors::ensure_constrained_finish;

use super::workflow::{WorkflowExecutionPlan, WorkflowNodeHost, WorkflowNodeRequest};
use super::{PipelineGenerateRequest, WorkflowRuntime};

/// The contracts a decode core implements for the interpreter.
///
/// One list, read both by the plan (which inputs the executors supply
/// themselves) and by the interpreter's node dispatch (which nodes to hand to
/// the host). A second copy of it could disagree with this one, and the symptom
/// would be a required input refused for a step that was never going to read
/// it.
pub(crate) const DECODE_CORE_CONTRACTS: &[&str] =
    &[AUTOREGRESSIVE_DECODE_CONTRACT, TOKEN_POLICY_CONTRACT];

/// Refuse a generation workflow this runtime cannot execute as declared.
///
/// Checking at load is what makes the declared document load-bearing: a package
/// whose loop says something the runtime would silently override fails here,
/// naming the field, rather than producing tokens that are the wrong tokens for
/// the workflow as written.
pub(crate) fn validate_generation_workflow(workflow: &WorkflowSpec) -> anyhow::Result<()> {
    let (body, continue_when, max_iterations, termination, carried) = workflow
        .steps
        .iter()
        .find_map(|step| match step {
            WorkflowStep::Loop {
                steps,
                continue_when,
                max_iterations,
                termination,
                carried,
                ..
            } => Some((steps, continue_when, max_iterations, termination, carried)),
            _ => None,
        })
        .context(
            "the workflow declares no loop; an autoregressive decoder's token stream is a loop \
             by construction",
        )?;

    anyhow::ensure!(
        *termination == onnx_genai_metadata::WorkflowLoopTermination::GenerationEos,
        "this workflow's loop declares termination '{termination:?}', but the runtime's token \
         policy ends generation; a predicate-terminated loop states its stop condition in the \
         graph and is executed by the interpreter's own predicate instead"
    );

    // The bound must be the request's, because the token policy bounds by the
    // request. A bound the runtime cannot see is a bound it would ignore.
    let bound = workflow.inputs.get(max_iterations).with_context(|| {
        format!(
            "this workflow's loop is bounded by '{max_iterations}', which it does not declare as \
             an input; the runtime bounds generation by the request's max_new_tokens and cannot \
             honour a bound it cannot see"
        )
    })?;
    anyhow::ensure!(
        matches!(
            &bound.role,
            onnx_genai_metadata::SemanticInputRole::Runtime { role, .. }
                if *role == onnx_genai_metadata::RuntimeInputRole::MaxIterations
                    || *role == onnx_genai_metadata::RuntimeInputRole::MaxOutputTokens
        ),
        "this workflow's loop is bounded by '{max_iterations}', which declares no max_iterations \
         role; the runtime would bound it by the request instead, silently ignoring the \
         package's own bound"
    );

    let serving = workflow.serving.as_ref().context(
        "this workflow declares a generation loop but no serving contract, so nothing names the \
         cell its liveness predicate reads",
    )?;
    anyhow::ensure!(
        continue_when == &serving.active,
        "this workflow's loop continues while '{continue_when}', but its serving contract names \
         '{}' as the active cell; the runtime's token policy writes the serving cell, so a \
         different predicate would never be updated",
        serving.active
    );
    for carry in carried {
        anyhow::ensure!(
            workflow.state.contains_key(&carry.cell),
            "this workflow's loop carries '{}', which it does not declare as state",
            carry.cell
        );
    }

    let mut decode_nodes = 0usize;
    let mut policy_nodes = 0usize;
    let mut emits_tokens = false;
    for step in body {
        match step {
            WorkflowStep::Invoke { component, .. } => {
                let declaration = workflow.components.get(component).with_context(|| {
                    format!("the loop invokes undeclared component '{component}'")
                })?;
                let contract = declaration.contract.as_ref().map(|c| c.id.as_str());
                match contract {
                    Some(AUTOREGRESSIVE_DECODE_CONTRACT) => decode_nodes += 1,
                    Some(TOKEN_POLICY_CONTRACT) => policy_nodes += 1,
                    Some(other) => anyhow::bail!(
                        "the generation loop invokes component '{component}' with contract \
                         '{other}', which no registered executor implements"
                    ),
                    None => anyhow::bail!(
                        "component '{component}' runs inside the generation loop but declares no \
                         contract, so this runtime cannot tell what step it is being asked to run"
                    ),
                }
            }
            WorkflowStep::Emit { output, .. } => {
                anyhow::ensure!(
                    workflow.outputs.contains_key(output),
                    "the loop emits into '{output}', which the workflow does not declare as an \
                     output"
                );
                emits_tokens |= output == onnx_genai_metadata::decoder_workflow::TOKENS_OUTPUT;
            }
            other => anyhow::bail!(
                "the generation loop body may only invoke components and emit; found {other:?}"
            ),
        }
    }
    anyhow::ensure!(
        decode_nodes == 1,
        "the generation loop runs {decode_nodes} autoregressive-decode steps; the decode core \
         executes exactly one forward pass per iteration"
    );
    anyhow::ensure!(
        policy_nodes == 1,
        "the generation loop applies {policy_nodes} token policies, so nothing would select or \
         stop exactly once per iteration"
    );
    anyhow::ensure!(
        emits_tokens,
        "the generation loop never emits '{}', so it would decode without producing the token \
         stream the caller reads",
        onnx_genai_metadata::decoder_workflow::TOKENS_OUTPUT
    );

    validate_runtime_managed_values(workflow, body)?;
    Ok(())
}

/// Refuse a workflow that reads a value the decode executors keep internal.
///
/// The fused decode session's logits stay where it produced them — on device,
/// behind a captured graph — and the token policy consumes them without the
/// interpreter ever seeing a tensor. That is only sound while the *only* reader
/// of the decode node's outputs is another runtime-implemented step. A package
/// routing logits into an ONNX component (an in-graph sampler, a reward head)
/// is asking for a value nobody would produce, and it must say so at load
/// rather than fail mid-generation.
fn validate_runtime_managed_values(
    workflow: &WorkflowSpec,
    body: &[WorkflowStep],
) -> anyhow::Result<()> {
    let mut runtime_managed = HashSet::new();
    for step in body {
        let WorkflowStep::Invoke {
            component, outputs, ..
        } = step
        else {
            continue;
        };
        let Some(declaration) = workflow.components.get(component) else {
            continue;
        };
        let is_decode = declaration
            .contract
            .as_ref()
            .is_some_and(|contract| contract.id == AUTOREGRESSIVE_DECODE_CONTRACT);
        if !is_decode {
            continue;
        }
        for (port, value) in outputs {
            // State the runtime already manages (the KV service group) never
            // becomes an SSA value in the first place; only ordinary outputs
            // are at issue here.
            if workflow
                .state
                .values()
                .any(|cell| cell.service_group.is_some() && cell.initializer == *value)
            {
                continue;
            }
            let _ = port;
            runtime_managed.insert(value.clone());
        }
    }
    for step in body {
        let WorkflowStep::Invoke {
            component, inputs, ..
        } = step
        else {
            continue;
        };
        let Some(declaration) = workflow.components.get(component) else {
            continue;
        };
        let hosted = declaration
            .contract
            .as_ref()
            .is_some_and(|contract| DECODE_CORE_CONTRACTS.contains(&contract.id.as_str()));
        if hosted {
            continue;
        }
        for value in inputs.values() {
            anyhow::ensure!(
                !runtime_managed.contains(value),
                "component '{component}' reads '{value}', which the autoregressive-decode \
                 executor keeps on the device it produced it on; a package that observes the \
                 decoder's scores must declare a component this runtime executes generically, \
                 not one bound to a runtime contract"
            );
        }
    }
    Ok(())
}

/// Package inputs the decode core's executors supply themselves.
///
/// A declared input read only by a step the runtime implements is that step's
/// business: the fused decode session builds its own attention mask and
/// position ids from the sequence it owns. Demanding the caller supply them
/// too would be asking for a second answer to a question the executor has
/// already answered — and the two could disagree.
///
/// Derived from what the workflow declares and which contracts the caller says
/// it implements. An input any generically executed component reads stays
/// required, so a composite package loses nothing.
pub(crate) fn host_supplied_inputs(workflow: &WorkflowSpec, hosted: &[&str]) -> HashSet<String> {
    let mut consumers: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    let mut visit = |steps: &[WorkflowStep]| {
        for step in steps {
            if let WorkflowStep::Invoke {
                component, inputs, ..
            } = step
            {
                let hosted_component = workflow
                    .components
                    .get(component)
                    .and_then(|declaration| declaration.contract.as_ref())
                    .is_some_and(|contract| hosted.contains(&contract.id.as_str()));
                for value in inputs.values() {
                    let Some((name, _)) = workflow.inputs.get_key_value(value) else {
                        continue;
                    };
                    let entry = consumers.entry(name.as_str()).or_default();
                    entry.0 += 1;
                    entry.1 += usize::from(hosted_component);
                }
            }
        }
    };
    visit(&workflow.steps);
    for step in &workflow.steps {
        if let WorkflowStep::Loop { setup, steps, .. } = step {
            visit(setup);
            visit(steps);
        }
    }
    consumers
        .into_iter()
        .filter(|(_, (total, hosted))| *total > 0 && total == hosted)
        .map(|(name, _)| name.to_string())
        .collect()
}

/// Everything a decode core contributes to a generation beyond executing nodes.
///
/// The interpreter produces the token stream; a core that owns paged KV, a
/// sampler and a session knows things the interpreter cannot see — why it
/// stopped, how much of the prompt a shared prefix already held, the per-token
/// logprobs it captured while scoring. A package with no such core simply has
/// none of those, and says so, rather than having them fabricated.
pub(crate) trait GenerationCore: WorkflowNodeHost {
    /// Why this core ended generation, if it did.
    fn finish(&self) -> Option<FinishReason>;
    /// Prompt tokens served from an already-populated cache.
    fn prefix_cache_hit_len(&self) -> usize;
    /// Per-token logprobs, when the request asked for them.
    fn logprobs(&self) -> Option<Vec<crate::config::TokenLogprob>>;
    /// The tokens this core committed, cross-checked against the emitted stream.
    fn committed_tokens(&self) -> Vec<crate::TokenId>;
    /// Whether this core already delivered each token to the request callback.
    fn streams_tokens(&self) -> bool;
}

/// Everything one generation needs beyond its decode executor.
pub(crate) struct GenerationRequest<'a> {
    pub(crate) options: &'a GenerateOptions,
    pub(crate) chain: &'a ProcessorChain,
    pub(crate) tokenizer: &'a Tokenizer,
    pub(crate) max_context: Option<usize>,
}

/// The decode core, as node executors the interpreter drives.
///
/// Advances exactly one declared node per call and holds nothing about the
/// loop. That split is what lets the fused decode session — paged KV, device
/// sampling, captured graphs — be a *step inside* a workflow rather than a
/// second loop beside one.
pub(crate) struct GenerationNodeHost<'a, 'c, B: DecodeLoopBackend + ?Sized> {
    pub(crate) backend: &'a mut B,
    pub(crate) state: &'a mut DecodeLoopState,
    options: &'a GenerateOptions,
    chain: &'a ProcessorChain,
    tokenizer: &'a Tokenizer,
    max_context: Option<usize>,
    callback: Option<&'a mut GenerateTokenCallback<'c>>,
    /// Logits from this iteration's decode node, awaiting its policy node.
    pending_forward: Option<ForwardOutcome>,
    /// The stop the policy node reached, if any.
    finish: Option<FinishReason>,
    /// Nodes this host executed, by contract. Proves — to a test, and to a
    /// reader — that the interpreter selected these executors from what the
    /// workflow declared rather than the drive calling them directly.
    executed: BTreeMap<&'static str, usize>,
}

impl<'a, 'c, B: DecodeLoopBackend + ?Sized> GenerationNodeHost<'a, 'c, B> {
    pub(crate) fn new(
        backend: &'a mut B,
        state: &'a mut DecodeLoopState,
        request: &GenerationRequest<'a>,
        callback: Option<&'a mut GenerateTokenCallback<'c>>,
    ) -> Self {
        Self {
            backend,
            state,
            options: request.options,
            chain: request.chain,
            tokenizer: request.tokenizer,
            max_context: request.max_context,
            callback,
            pending_forward: None,
            finish: None,
            executed: BTreeMap::new(),
        }
    }

    /// The stop this core reached, if its policy node reached one.
    pub(crate) fn reached_finish(&self) -> Option<FinishReason> {
        self.finish.clone()
    }

    fn run_decode_node(&mut self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.pending_forward.is_none(),
            "the generation loop ran two decoder forward passes without applying its token \
             policy between them; the second pass's scores would replace the first's unread"
        );
        self.pending_forward = Some(forward_step(
            self.backend,
            self.state,
            self.options,
            self.chain,
        )?);
        Ok(())
    }

    fn run_policy_node(&mut self, request: &mut WorkflowNodeRequest<'_>) -> anyhow::Result<()> {
        let forward = self.pending_forward.take().context(
            "the generation loop applied its token policy before any decoder forward pass \
             produced logits",
        )?;
        let (token, mut reason) = select_and_commit_step(
            self.backend,
            self.state,
            self.options,
            self.chain,
            self.tokenizer,
            forward,
            self.callback.as_deref_mut(),
        )?;
        // The context bound is a stop like any other, decided by the policy that
        // just committed a token: the next iteration is the one that could not
        // fit. Expressing it here rather than as a branch around the loop is
        // what lets `continue_when` remain the single liveness predicate.
        if reason.is_none() && reached_context_limit(self.backend.context_len(), self.max_context) {
            reason = Some(FinishReason::Length);
        }
        self.finish = reason.clone();
        self.publish_policy_outputs(request, token, reason.is_some())
    }

    /// Define the SSA values the policy node declares.
    ///
    /// The interpreter reads these and nothing else: `active` is the liveness
    /// predicate the loop tests, and the token is what the emit publishes. A
    /// policy that committed a token without defining them would leave the loop
    /// running on last iteration's answer.
    fn publish_policy_outputs(
        &self,
        request: &mut WorkflowNodeRequest<'_>,
        token: crate::TokenId,
        finished: bool,
    ) -> anyhow::Result<()> {
        for (port, value) in request.outputs.iter() {
            let tensor = match port.as_str() {
                "token" => Value::from_slice_i64(&[i64::from(token)], &[1, 1])?,
                "active" => bool_row(!finished)?,
                "done" => bool_row(finished)?,
                "accepted_len" => Value::from_slice_i64(&[1], &[1])?,
                // A static-cache package carries each row's logical cache
                // length; the executor that owns the cache is the only thing
                // that knows it.
                "cache_lengths" | "lengths" => Value::from_slice_i64(
                    &[i64::try_from(self.backend.context_len())
                        .context("cache length exceeds int64")?],
                    &[1],
                )?,
                other => anyhow::bail!(
                    "the token policy declares output port '{other}', which this runtime's \
                     policy executor does not produce"
                ),
            };
            request.values.insert(value.clone(), tensor);
        }
        Ok(())
    }
}

fn bool_row(value: bool) -> anyhow::Result<Value> {
    Value::from_raw_bytes(vec![u8::from(value)], &[1], DataType::Bool).map_err(Into::into)
}

impl<B: DecodeLoopBackend + ?Sized> WorkflowNodeHost for GenerationNodeHost<'_, '_, B> {
    fn hosted_contracts(&self) -> &'static [&'static str] {
        DECODE_CORE_CONTRACTS
    }

    fn execute_contract_node(
        &mut self,
        mut request: WorkflowNodeRequest<'_>,
    ) -> anyhow::Result<bool> {
        match request.contract {
            AUTOREGRESSIVE_DECODE_CONTRACT => {
                self.run_decode_node()?;
                *self
                    .executed
                    .entry(AUTOREGRESSIVE_DECODE_CONTRACT)
                    .or_default() += 1;
                Ok(true)
            }
            TOKEN_POLICY_CONTRACT => {
                self.run_policy_node(&mut request)?;
                *self.executed.entry(TOKEN_POLICY_CONTRACT).or_default() += 1;
                Ok(true)
            }
            _ => Ok(false),
        }
    }
}

impl<B: DecodeLoopBackend + ?Sized> GenerationCore for GenerationNodeHost<'_, '_, B> {
    fn finish(&self) -> Option<FinishReason> {
        self.reached_finish()
    }

    fn prefix_cache_hit_len(&self) -> usize {
        self.state.prefix_cache_hit_len
    }

    fn logprobs(&self) -> Option<Vec<crate::config::TokenLogprob>> {
        self.state.logprobs.clone()
    }

    fn committed_tokens(&self) -> Vec<crate::TokenId> {
        self.state.generated_tokens.clone()
    }

    fn streams_tokens(&self) -> bool {
        true
    }
}

/// Run a package's declared generation loop to completion.
///
/// The one drive. The loop bound, the liveness predicate, the carried state and
/// the emit that publishes the token stream are the workflow's, walked by the
/// interpreter. `core` supplies executors for the steps the package declares as
/// runtime contracts; a package that declares none — one whose sampler and
/// termination predicate are ONNX components — passes `None` and the same
/// interpreter runs every step from its artifact.
pub(crate) fn run_declared_generation(
    runtime: &WorkflowRuntime,
    options: &GenerateOptions,
    tokenizer: Option<&Tokenizer>,
    request: PipelineGenerateRequest,
    mut core: Option<&mut dyn GenerationCore>,
    mut callback: Option<&mut GenerateTokenCallback<'_>>,
) -> anyhow::Result<GenerateResult> {
    let hosted: &[&str] = match core.as_deref() {
        Some(core) => core.hosted_contracts(),
        None => &[],
    };
    let mut plan = WorkflowExecutionPlan::new_hosted(runtime, request, hosted)?;
    let outputs = {
        let mut host: Option<&mut dyn WorkflowNodeHost> = core
            .as_deref_mut()
            .map(|core| core as &mut dyn WorkflowNodeHost);
        let (values, row_outputs) = plan.execute_retained_with_host(&mut host)?;
        runtime.package_outputs(values, row_outputs)?
    };

    let output = runtime
        .workflow_spec()
        .outputs
        .iter()
        .find(|(_, output)| output.role == onnx_genai_metadata::WorkflowOutputRole::Tokens)
        .map(|(name, _)| name.clone())
        .context("generation requires one package output with role: tokens")?;
    let rows = outputs.rows(&output);
    anyhow::ensure!(
        rows.len() <= 1,
        "generation cannot flatten multi-row ragged output '{output}'; consume semantic row \
         streams with the tensor workflow API instead"
    );
    let token_ids = outputs
        .aggregate(&output)
        .or_else(|| rows.first().map(|(_, value)| *value))
        .with_context(|| format!("the workflow did not emit tokens output '{output}'"))?
        .to_vec_i64()?
        .into_iter()
        .map(|token| u32::try_from(token).context("the workflow emitted a token outside uint32"))
        .collect::<anyhow::Result<Vec<_>>>()?;

    // The emitted stream and the core's committed history describe the same
    // generation. If they disagree, one is wrong and nothing outside can tell
    // which — so this refuses rather than picking one.
    if let Some(core) = core.as_deref() {
        assert_streams_agree(&output, &token_ids, &core.committed_tokens())?;
    }

    // Why the loop stopped, from whatever knows: a decode core that reached a
    // stop, or the interpreter's own record of whether the liveness predicate
    // ended the loop. Reporting `MaxTokens` unconditionally would tell a caller
    // that a workflow which ended at its own EOS ran out of budget, which is
    // the one thing a finish reason exists to distinguish.
    let finish_reason = match core.as_deref().and_then(GenerationCore::finish) {
        Some(reason) => reason,
        None if runtime.last_generation_ended_by_predicate() => FinishReason::EosToken,
        None => FinishReason::MaxTokens,
    };

    let text = tokenizer
        .map(|tokenizer| tokenizer.decode(&token_ids))
        .transpose()?
        .unwrap_or_default();
    ensure_constrained_finish(options, &text, finish_reason.clone())?;

    // A core that scored each token already delivered it. Replaying the stream
    // here would double every token a server streamed.
    let already_streamed = core.as_deref().is_some_and(GenerationCore::streams_tokens);
    if let (Some(callback), false) = (callback.as_mut(), already_streamed) {
        for (index, token_id) in token_ids.iter().copied().enumerate() {
            let token_text = tokenizer
                .map(|tokenizer| tokenizer.decode(&[token_id]))
                .transpose()?
                .unwrap_or_default();
            callback(crate::config::GenerateToken {
                token_id,
                text: token_text,
                finish_reason: (index + 1 == token_ids.len()).then(|| finish_reason.clone()),
            })?;
        }
    }

    Ok(GenerateResult {
        text,
        token_ids,
        finish_reason,
        // A package with no decode core has no prefix cache to have hit: the
        // interpreter recomputes every component from the request's own inputs.
        prefix_cache_hit_len: core
            .as_deref()
            .map_or(0, GenerationCore::prefix_cache_hit_len),
        // Logprobs are the scoring core's record. A package whose sampler is an
        // ONNX component never scored on the host, so it has none to report —
        // which is a fact about the package, not a missing feature.
        logprobs: core.as_deref().and_then(GenerationCore::logprobs),
        budget_cap: None,
    })
}

/// Generate through a decode core executing the package's declared loop.
///
/// The direct replacement for a hand-rolled token loop: the iteration bound,
/// the stop predicate and the emit come from the workflow, and `backend` is
/// the executor the interpreter routes the declared decode step to.
pub(crate) fn generate_with_decode_core<B: DecodeLoopBackend + ?Sized>(
    runtime: &WorkflowRuntime,
    backend: &mut B,
    state: &mut DecodeLoopState,
    prompt_tokens: &[crate::TokenId],
    request: GenerationRequest<'_>,
    callback: Option<&mut GenerateTokenCallback<'_>>,
) -> anyhow::Result<GenerateResult> {
    // A prompt that already fills the context cannot take a step. Refusing
    // before the plan binds anything keeps the refusal free of partial state.
    if reached_context_limit(backend.context_len(), request.max_context) {
        ensure_constrained_finish(request.options, &state.generated_text, FinishReason::Length)?;
        return finish_result(
            request.tokenizer,
            &state.generated_tokens,
            FinishReason::Length,
            state.prefix_cache_hit_len,
            state.logprobs.as_deref(),
        );
    }
    state.generated_tokens.reserve(
        request
            .options
            .max_new_tokens
            .saturating_sub(state.generated_tokens.len()),
    );
    if let Some(logprobs) = state.logprobs.as_mut() {
        logprobs.reserve(
            request
                .options
                .max_new_tokens
                .saturating_sub(logprobs.len()),
        );
    }
    let options = request.options.clone();
    let tokenizer = request.tokenizer;
    let pipeline_request = PipelineGenerateRequest::new(crate::GenerateRequest {
        prompt: crate::GeneratePrompt::TokenIds(prompt_tokens.to_vec()),
        options: options.clone(),
    });
    let mut host = GenerationNodeHost::new(backend, state, &request, callback);
    run_declared_generation(
        runtime,
        &options,
        Some(tokenizer),
        pipeline_request,
        Some(&mut host),
        None,
    )
}

/// The declared token stream and the generated one must be the same stream.
///
/// The workflow's `emit` is what publishes tokens to a caller reading the
/// package's output; the decode core's committed history is what the text and
/// logprobs are built from. Checking them against each other is what keeps the
/// emit load-bearing rather than decorative — an emit that silently dropped or
/// duplicated a token would otherwise be invisible.
pub(crate) fn verify_emitted_tokens(
    runtime: &WorkflowRuntime,
    cursor: &super::WorkflowGenerationCursor,
    committed: &[crate::TokenId],
) -> anyhow::Result<()> {
    let emitted = cursor
        .emitted_tokens(runtime)?
        .into_iter()
        .map(|token| u32::try_from(token).context("the workflow emitted a token outside uint32"))
        .collect::<anyhow::Result<Vec<_>>>()?;
    assert_streams_agree("tokens", &emitted, committed)
}

fn assert_streams_agree(
    output: &str,
    emitted: &[crate::TokenId],
    committed: &[crate::TokenId],
) -> anyhow::Result<()> {
    anyhow::ensure!(
        emitted == committed,
        "the workflow emitted {emitted:?} into '{output}' while the decode core committed \
         {committed:?}; the declared token stream and the generated one must be the same stream"
    );
    Ok(())
}

/// Package inputs that exist only to seed state a service owns.
///
/// A cell declaring `management: runtime` says its buffer belongs to the
/// service that runs it — the paged KV cache, the fused session's captured
/// workspace. Its declared initializer names where a *generic* execution would
/// get an empty starting tensor, and for a growing cache that tensor has no
/// shape the interpreter could build: rank is declared, extent is not.
///
/// Materializing one would mean inventing a buffer the executor is about to
/// replace, and failing to materialize one would refuse a package that is
/// perfectly well formed. So an input that seeds only runtime-managed cells,
/// and that no executed step reads, is left unbound: the service supplies the
/// buffer, exactly as the package says it does.
pub(crate) fn runtime_managed_seeds(workflow: &WorkflowSpec) -> HashSet<String> {
    let mut consumed = HashSet::new();
    let mut visit = |steps: &[WorkflowStep]| {
        for step in steps {
            if let WorkflowStep::Invoke { inputs, .. } = step {
                consumed.extend(inputs.values().cloned());
            }
        }
    };
    visit(&workflow.steps);
    for step in &workflow.steps {
        if let WorkflowStep::Loop { setup, steps, .. } = step {
            visit(setup);
            visit(steps);
        }
    }
    let mut seeds: HashSet<String> = HashSet::new();
    let mut non_runtime_seeds: HashSet<String> = HashSet::new();
    for cell in workflow.state.values() {
        if cell.management == onnx_genai_metadata::StateManagement::Runtime {
            seeds.insert(cell.initializer.clone());
        } else {
            non_runtime_seeds.insert(cell.initializer.clone());
        }
    }
    seeds
        .into_iter()
        .filter(|seed| {
            workflow.inputs.contains_key(seed)
                && !consumed.contains(seed)
                && !non_runtime_seeds.contains(seed)
        })
        .collect()
}

/// A hosted interpreter over a canonical decoder workflow, for unit tests.
///
/// The loop a unit test drives must be a *declared* loop, or the test proves
/// nothing about what a package gets. Building it from the same emitter a real
/// minimal decoder goes through keeps these cases on the production path
/// instead of a bespoke shape that could diverge from it.
#[cfg(test)]
pub(crate) fn test_decoder_runtime() -> anyhow::Result<WorkflowRuntime> {
    let abi = onnx_genai_metadata::DecoderAbi {
        token_input: Some("input_ids".to_string()),
        logits_output: Some("logits".to_string()),
        kv_inputs: Some(vec!["past.key".to_string(), "past.value".to_string()]),
        kv_outputs: Some(vec!["present.key".to_string(), "present.value".to_string()]),
        ..onnx_genai_metadata::DecoderAbi::default()
    };
    let workflow = onnx_genai_metadata::decoder_workflow::decoder_workflow(
        &abi,
        "model.onnx",
        &onnx_genai_metadata::decoder_workflow::DecoderFacts::default(),
    )
    .map_err(|error| {
        anyhow::anyhow!("a minimal decoder is expressible as a workflow: {error:?}")
    })?;
    validate_generation_workflow(&workflow)?;
    let directory = onnx_genai_ort::PipelineModelDirectory {
        root: std::path::PathBuf::from("."),
        metadata_path: None,
        spec: onnx_genai_metadata::PipelineSpec {
            workflow: workflow.clone(),
        },
        adapters: None,
        metadata: None,
        preprocessing: None,
        model_paths: BTreeMap::new(),
        tokenizer_paths: onnx_genai_ort::PipelineTokenizerPaths {
            shared: None,
            per_component: BTreeMap::new(),
        },
    };
    WorkflowRuntime::hosted(
        std::path::PathBuf::from("."),
        workflow,
        crate::EngineDecodeBackend::Ort,
        crate::MemoryStrategyPlan::unknown(0, None, "hosted test interpreter"),
        onnx_genai_ort::PipelineModels::hosted(
            directory,
            onnx_genai_ort::SessionOptions::default(),
            None,
        ),
        None,
    )
}

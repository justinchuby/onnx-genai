//! Universal typed workflow interpreter.

use std::collections::BTreeMap;

use super::*;
use crate::decode::clone_value;
use onnx_genai_metadata::{StateAliasing, StatePortAccess};
use onnx_genai_ort::{IoBinding, Session};

type ResolvedComponentInvocation<'a> = (
    &'a str,
    &'a onnx_genai_metadata::WorkflowComponent,
    std::collections::BTreeMap<String, String>,
    std::collections::BTreeMap<String, String>,
);

struct ActiveAdapterContextGuard<'a>(
    &'a std::cell::RefCell<Option<super::adapters::AdapterRunContext>>,
);

impl Drop for ActiveAdapterContextGuard<'_> {
    fn drop(&mut self) {
        self.0.borrow_mut().take();
    }
}

/// Prefix of the error a workflow raises when a required package input was
/// never supplied.
///
/// A front end turns this into advice about the attachment the caller most
/// likely forgot, so the wording is shared rather than matched by hand: the
/// message and its recognizer cannot drift apart if they name the same
/// constant.
pub const MISSING_REQUIRED_INPUT: &str = "required workflow package input ";

/// Prefix of the error a workflow raises when a session's conversation would
/// exceed the bound the package declares for it.
///
/// A front end turns this into a request-level refusal rather than a server
/// fault: the caller can shorten the turn, ask for fewer tokens, or start a new
/// session, and none of those are things the server did wrong.
pub const CONVERSATION_OVER_BOUND: &str = "session conversation exceeds its declared bound";

/// True when `error`, or anything it wraps, is a conversation over its bound.
pub fn is_conversation_over_bound(error: &anyhow::Error) -> bool {
    error
        .chain()
        .any(|cause| cause.to_string().contains(CONVERSATION_OVER_BOUND))
}

/// True when `error`, or anything it wraps, is a required package input the
/// request never supplied.
pub fn is_missing_required_input(error: &anyhow::Error) -> bool {
    error
        .chain()
        .any(|cause| cause.to_string().starts_with(MISSING_REQUIRED_INPUT))
}

#[cfg(test)]
mod missing_input_tests {
    use super::*;

    #[test]
    fn a_missing_required_input_is_recognized_through_the_context_wrapped_around_it() {
        let raw =
            anyhow::anyhow!("{MISSING_REQUIRED_INPUT}'encoder.pixel_values' was not supplied");
        assert!(is_missing_required_input(&raw));
        assert!(is_missing_required_input(
            &raw.context("decoder forward failed")
        ));
    }

    #[test]
    fn an_unrelated_failure_is_not_mistaken_for_a_missing_input() {
        let oom = anyhow::anyhow!("cuda_ep: cuMemAlloc: CUDA_ERROR_OUT_OF_MEMORY")
            .context("decoder forward failed");
        assert!(!is_missing_required_input(&oom));

        // An input the request supplied but the workflow never declared is a
        // caller/package mismatch, not a forgotten attachment, so it must not be
        // advised as one.
        let undeclared = anyhow::anyhow!(
            "workflow request supplied undeclared application inputs: [\"encoder.pixel_values\"]"
        );
        assert!(!is_missing_required_input(&undeclared));
    }
}

pub(crate) fn compile_device_bridge_components(
    graph: &WorkflowNode,
    island_components: &HashSet<String>,
) -> HashSet<String> {
    fn collect(
        node: &WorkflowNode,
        recurring: bool,
        producers: &mut HashMap<String, String>,
        invocations: &mut Vec<(String, Vec<String>, bool)>,
    ) {
        match node {
            WorkflowNode::Sequence { nodes } => {
                for node in nodes {
                    collect(node, recurring, producers, invocations);
                }
            }
            WorkflowNode::Invoke {
                component,
                inputs,
                outputs,
                ..
            } => {
                for value in outputs.values() {
                    producers.insert(value.clone(), component.clone());
                }
                invocations.push((
                    component.clone(),
                    inputs.values().cloned().collect::<Vec<_>>(),
                    recurring,
                ));
            }
            WorkflowNode::Loop { setup, body, .. } => {
                collect(setup, recurring, producers, invocations);
                collect(body, true, producers, invocations);
            }
            WorkflowNode::Branch { cases, default, .. } => {
                for case in cases.values() {
                    collect(case, recurring, producers, invocations);
                }
                if let Some(default) = default {
                    collect(default, recurring, producers, invocations);
                }
            }
            WorkflowNode::Emit { .. }
            | WorkflowNode::Transfer { .. }
            | WorkflowNode::ExecutionIsland { .. } => {}
        }
    }

    let mut producers = HashMap::new();
    let mut invocations = Vec::new();
    collect(graph, false, &mut producers, &mut invocations);
    let recurring_components = invocations
        .iter()
        .filter_map(|(component, _, recurring)| recurring.then_some(component.clone()))
        .collect::<HashSet<_>>();
    let mut needed = island_components.clone();
    let mut frontier = island_components.clone();
    for _ in 0..2 {
        let mut next = HashSet::new();
        for (index, (component, inputs, _)) in invocations.iter().enumerate() {
            if frontier.contains(component) {
                let mut unresolved = false;
                for input in inputs {
                    if let Some(producer) = producers.get(input) {
                        next.insert(producer.clone());
                    } else {
                        unresolved = true;
                    }
                }
                if unresolved
                    && let Some((previous, _, _)) =
                        index.checked_sub(1).and_then(|i| invocations.get(i))
                {
                    next.insert(previous.clone());
                }
            }
        }
        next.retain(|component| !needed.contains(component));
        if next.is_empty() {
            break;
        }
        needed.extend(next.iter().cloned());
        frontier = next;
    }
    needed.retain(|component| !island_components.contains(component));
    needed.retain(|component| recurring_components.contains(component));
    needed
}

pub(crate) type ComponentBindingKey = (String, Vec<(String, Vec<i64>)>);
pub(crate) type ComponentOutputKey = (String, String, Vec<i64>, String);

pub(crate) struct StableComponentBinding {
    binding: IoBinding,
    inputs: Vec<(String, Arc<Value>)>,
    outputs: Vec<(String, Arc<Value>)>,
    shared_outputs: Vec<(String, usize)>,
    output_order: Vec<String>,
    service_generation: u64,
    graph_id: i32,
    captured: bool,
    _allocator: Arc<onnx_genai_ort::Allocator>,
}

fn stable_component_outputs(
    stable: &StableComponentBinding,
) -> anyhow::Result<Vec<(String, Value)>> {
    stable
        .output_order
        .iter()
        .map(|name| {
            let value = if let Some((_, value)) =
                stable.outputs.iter().find(|(output, _)| output == name)
            {
                value
            } else {
                let input_index = match stable
                    .shared_outputs
                    .iter()
                    .find(|(output, _)| output == name)
                {
                    Some((_, input_index)) => *input_index,
                    None => {
                        return Err(anyhow::anyhow!(
                            "stable component output '{name}' is unavailable"
                        ));
                    }
                };
                &stable.inputs[input_index].1
            };
            Value::alias_from_shared_owner(Arc::clone(value), value.shape())
                .map(|value| (name.clone(), value))
                .map_err(Into::into)
        })
        .collect()
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct WorkflowPerformanceDiagnostic {
    pub runs: u64,
    pub total_elapsed_ns: u128,
    pub last_elapsed_ns: u128,
    pub last_ttft_ns: Option<u128>,
    pub last_emit_timestamps_ns: Vec<u128>,
    pub last_loop_iterations: u64,
    pub last_component_invocations: u64,
    pub last_stage_runs: BTreeMap<String, u64>,
    pub last_stage_elapsed_ns: BTreeMap<String, u128>,
    pub last_emit_events: u64,
    pub last_emitted_elements: u64,
    pub last_steps_per_second: f64,
    pub last_elements_per_second: f64,
    pub islands: Vec<ExecutionIslandDiagnostic>,
}

#[derive(Default)]
pub(crate) struct WorkflowPerformanceCounters {
    runs: u64,
    total_elapsed_ns: u128,
    last_elapsed_ns: u128,
    last_ttft_ns: Option<u128>,
    last_emit_timestamps_ns: Vec<u128>,
    last_loop_iterations: u64,
    last_component_invocations: u64,
    last_stage_runs: BTreeMap<String, u64>,
    last_stage_elapsed_ns: BTreeMap<String, u128>,
    last_emit_events: u64,
    last_emitted_elements: u64,
    last_loop_ended_by_predicate: bool,
}

/// Persistent execution state for repeated runs of one compiled workflow.
///
/// Preparing once caches request/literal binding, input validation, symbol
/// resolution, component override validation, and the input value slots. Each
/// execution retains those inputs while discarding transient SSA and emit values.
pub struct WorkflowExecutionPlan<'a> {
    engine: &'a WorkflowRuntime,
    values: PipelineTensors,
    input_names: Vec<String>,
    input_aliases: HashMap<String, String>,
    initial_symbols: HashMap<String, i64>,
    dynamic_symbols: std::collections::HashSet<String>,
    session_id: Option<String>,
    component_overrides: HashMap<String, String>,
    max_iterations_only: bool,
    /// Tokens this pass's prompt input carries, when a declared continuation
    /// made them the conversation rather than just this turn's request.
    ///
    /// Kept because the cell this pass publishes is the whole conversation, and
    /// the emitted stream alone is only its last turn.
    continued_prompt: Option<Vec<i64>>,
}

#[derive(Default)]
pub(crate) struct WorkflowRunTelemetry {
    started: Option<std::time::Instant>,
    first_emit_ns: Option<u128>,
    emit_timestamps_ns: Vec<u128>,
    max_iterations_only: bool,
    loop_iterations: u64,
    /// Whether the loop that produces the package's token stream ended because
    /// its liveness predicate went false, rather than exhausting its bound.
    ///
    /// This is *why generation stopped*, and it is only knowable here: the loop
    /// is the thing that decided. Without it a caller has to guess, and the
    /// guess that was being made was "always the bound", so a workflow that
    /// stopped at EOS reported `MaxTokens`.
    loop_ended_by_predicate: bool,
    component_invocations: u64,
    emit_events: u64,
    emitted_elements: u64,
    stage_runs: BTreeMap<String, u64>,
    stage_elapsed_ns: BTreeMap<String, u128>,
    row_outputs: BTreeMap<String, Vec<String>>,
}

impl WorkflowRunTelemetry {
    fn record_stage(&mut self, name: impl Into<String>, elapsed_ns: u128) {
        let name = name.into();
        *self.stage_runs.entry(name.clone()).or_default() += 1;
        *self.stage_elapsed_ns.entry(name).or_default() += elapsed_ns;
    }
}

type WorkflowAdapterExecutor = fn(
    &WorkflowRuntime,
    &str,
    &std::collections::BTreeMap<String, String>,
    &std::collections::BTreeMap<String, String>,
    &onnx_genai_metadata::WorkflowComponent,
    &mut PipelineTensors,
    &HashMap<String, i64>,
    &mut HashMap<String, i64>,
) -> anyhow::Result<()>;

fn workflow_adapter_registry()
-> &'static HashMap<(&'static str, &'static str), WorkflowAdapterExecutor> {
    static REGISTRY: std::sync::LazyLock<
        HashMap<(&'static str, &'static str), WorkflowAdapterExecutor>,
    > = std::sync::LazyLock::new(|| {
        HashMap::from([
            (
                ("onnx-genai.image-preprocess", "1"),
                WorkflowRuntime::run_image_preprocess_adapter as WorkflowAdapterExecutor,
            ),
            (
                ("onnx-genai.audio-preprocess", "1"),
                WorkflowRuntime::run_audio_preprocess_adapter as WorkflowAdapterExecutor,
            ),
            (
                ("onnx-genai.grammar-guidance", "1"),
                WorkflowRuntime::run_grammar_guidance_adapter as WorkflowAdapterExecutor,
            ),
            (
                ("onnx-genai.telemetry", "1"),
                WorkflowRuntime::run_telemetry_adapter as WorkflowAdapterExecutor,
            ),
            (
                ("onnx-genai.parameter-overlay", "1"),
                WorkflowRuntime::run_parameter_overlay_adapter as WorkflowAdapterExecutor,
            ),
            (
                ("onnx-genai.text-assembly", "1"),
                WorkflowRuntime::run_text_assembly_adapter as WorkflowAdapterExecutor,
            ),
        ])
    });
    &REGISTRY
}

pub(super) fn supports_workflow_adapter(abi: &str, version: &str) -> bool {
    workflow_adapter_registry().contains_key(&(abi, version))
}

fn validate_component_overrides(
    workflow: &WorkflowSpec,
    overrides: &HashMap<String, String>,
) -> anyhow::Result<()> {
    for (target_name, replacement_name) in overrides {
        let target = workflow.components.get(target_name).with_context(|| {
            format!("component override target '{target_name}' is not declared by the package")
        })?;
        if !target.application_overridable {
            anyhow::bail!(
                "workflow component '{target_name}' does not allow application replacement"
            );
        }
        let replacement = workflow.components.get(replacement_name).with_context(|| {
            format!("replacement component '{replacement_name}' is not declared by the package")
        })?;
        if !matches!(target.implementation, ComponentImplementation::Onnx { .. })
            || !matches!(
                replacement.implementation,
                ComponentImplementation::Onnx { .. }
            )
        {
            anyhow::bail!(
                "component override '{target_name}' -> '{replacement_name}' requires ONNX components"
            );
        }
        let target_contract = target.contract.as_ref().with_context(|| {
            format!("overridable component '{target_name}' has no versioned contract")
        })?;
        let replacement_contract = replacement.contract.as_ref().with_context(|| {
            format!("replacement component '{replacement_name}' has no versioned contract")
        })?;
        if target_contract.id != replacement_contract.id
            || target_contract.version != replacement_contract.version
        {
            anyhow::bail!(
                "replacement component '{replacement_name}' has contract {}@{}, expected {}@{} \
                 for '{target_name}'",
                replacement_contract.id,
                replacement_contract.version,
                target_contract.id,
                target_contract.version
            );
        }
        if target_contract
            .bindings
            .keys()
            .collect::<std::collections::BTreeSet<_>>()
            != replacement_contract
                .bindings
                .keys()
                .collect::<std::collections::BTreeSet<_>>()
        {
            anyhow::bail!(
                "replacement component '{replacement_name}' does not implement the complete \
                 semantic port ABI of '{target_name}'"
            );
        }
        if target.effects != replacement.effects {
            anyhow::bail!(
                "replacement component '{replacement_name}' does not match the effect ABI of \
                 '{target_name}'"
            );
        }
        validate_replacement_port_contracts(target_name, target, replacement_name, replacement)?;
    }
    Ok(())
}

fn validate_replacement_port_contracts(
    target_name: &str,
    target: &onnx_genai_metadata::WorkflowComponent,
    replacement_name: &str,
    replacement: &onnx_genai_metadata::WorkflowComponent,
) -> anyhow::Result<()> {
    let target_bindings = &target
        .contract
        .as_ref()
        .with_context(|| format!("overridable component '{target_name}' has no contract"))?
        .bindings;
    let replacement_bindings = &replacement
        .contract
        .as_ref()
        .with_context(|| format!("replacement component '{replacement_name}' has no contract"))?
        .bindings;
    for (role, target_port) in target_bindings {
        let replacement_port = &replacement_bindings[role];
        let target_input = target.ports.inputs.get(target_port);
        let replacement_input = replacement.ports.inputs.get(replacement_port);
        let target_output = target.ports.outputs.get(target_port);
        let replacement_output = replacement.ports.outputs.get(replacement_port);
        if target_input.is_some() != replacement_input.is_some()
            || target_output.is_some() != replacement_output.is_some()
        {
            anyhow::bail!(
                "replacement component '{replacement_name}' semantic port '{role}' has a \
                 different direction from '{target_name}'"
            );
        }
        if let (Some(expected), Some(actual)) = (target_input, replacement_input)
            && expected != actual
        {
            anyhow::bail!(
                "replacement component '{replacement_name}' input '{replacement_port}' is \
                 incompatible with '{target_name}.{target_port}'"
            );
        }
        if let (Some(expected), Some(actual)) = (target_output, replacement_output)
            && expected != actual
        {
            anyhow::bail!(
                "replacement component '{replacement_name}' output '{replacement_port}' is \
                 incompatible with '{target_name}.{target_port}'"
            );
        }
    }
    Ok(())
}

fn resolve_component_invocation<'a>(
    workflow: &'a WorkflowSpec,
    component: &'a str,
    declaration: &'a onnx_genai_metadata::WorkflowComponent,
    inputs: &std::collections::BTreeMap<String, String>,
    outputs: &std::collections::BTreeMap<String, String>,
    overrides: &HashMap<String, String>,
) -> anyhow::Result<ResolvedComponentInvocation<'a>> {
    let Some(replacement_name) = overrides.get(component) else {
        return Ok((component, declaration, inputs.clone(), outputs.clone()));
    };
    let (replacement_name, replacement) = workflow
        .components
        .get_key_value(replacement_name)
        .with_context(|| format!("replacement component '{replacement_name}' is undeclared"))?;
    let target_bindings = &declaration
        .contract
        .as_ref()
        .with_context(|| format!("overridable component '{component}' has no contract"))?
        .bindings;
    let replacement_bindings = &replacement
        .contract
        .as_ref()
        .with_context(|| format!("replacement component '{replacement_name}' has no contract"))?
        .bindings;
    let remap = |ports: &std::collections::BTreeMap<String, String>| {
        ports
            .iter()
            .map(|(port, value)| {
                let role = target_bindings
                    .iter()
                    .find_map(|(role, bound)| (bound == port).then_some(role))
                    .with_context(|| {
                        format!(
                            "overridable component '{component}' invoked port '{port}' is not \
                             covered by its semantic contract ABI"
                        )
                    })?;
                let replacement_port = replacement_bindings.get(role).with_context(|| {
                    format!(
                        "replacement component '{replacement_name}' has no binding for semantic \
                         port '{role}'"
                    )
                })?;
                Ok((replacement_port.clone(), value.clone()))
            })
            .collect::<anyhow::Result<std::collections::BTreeMap<_, _>>>()
    };
    Ok((
        replacement_name.as_str(),
        replacement,
        remap(inputs)?,
        remap(outputs)?,
    ))
}

fn session_state_value_name(cell: &str) -> String {
    format!("__session_state.{cell}")
}

/// The one cell whose lease says a conversation rejoins the next invocation
/// through the prompt binding.
///
/// Validation admits at most one, so this returns the first rather than
/// choosing between two: a package with two conversations was refused before it
/// ever loaded.
pub(crate) fn workflow_prompt_continuation(
    workflow: &WorkflowSpec,
) -> Option<(&str, &str, &str, &onnx_genai_metadata::WorkflowStateCell)> {
    workflow.state.iter().find_map(|(cell, state)| {
        let onnx_genai_metadata::SessionContinuation::PromptPrefix {
            prompt_input,
            tokens_output,
        } = state.session.as_ref()?.continuation.as_ref()?;
        Some((
            cell.as_str(),
            prompt_input.as_str(),
            tokens_output.as_str(),
            state,
        ))
    })
}

/// The declared upper bound on a continuation cell's length.
///
/// The validator makes the bound mandatory on a continuation and requires it to
/// name a declared workflow input, so the value is bound before the pass starts
/// and is knowable both when a turn is admitted and when it completes. Reading
/// it here is what makes that statement load-bearing rather than decorative:
/// nothing else checks it, because a continuation is deliberately not
/// loop-carried and so never reaches the carry path's recurrence check.
///
/// A missing value is an error, not a skipped check. Returning "no bound" for a
/// symbol the document named would be the silent version of the failure this
/// exists to prevent.
fn continuation_bound(
    cell: &str,
    state: &onnx_genai_metadata::WorkflowStateCell,
    values: &PipelineTensors,
) -> anyhow::Result<usize> {
    let max = match &state.recurrence {
        onnx_genai_metadata::ShapeRecurrence::Growing { max, .. }
        | onnx_genai_metadata::ShapeRecurrence::Bounded { max, .. } => max,
        onnx_genai_metadata::ShapeRecurrence::Invariant => anyhow::bail!(
            "session continuation state '{cell}' declares no growth bound, but a conversation \
             grows with every turn"
        ),
    };
    // Per-row bounds would be several conversations; a session is one, so the
    // narrowest row is the bound the conversation has to keep.
    workflow_usize_rows(values, max)?
        .into_iter()
        .min()
        .with_context(|| {
            format!("session continuation state '{cell}' bound '{max}' carries no value")
        })
}

/// Read the tokens a session-scoped continuation cell holds, as a flat row.
fn continuation_tokens(value: &Value) -> anyhow::Result<Vec<i64>> {
    let shape = value.shape();
    let rows = if shape.len() >= 2 { shape[0] } else { 1 };
    anyhow::ensure!(
        rows == 1,
        "a session continues one conversation, but its state holds {rows} batch rows"
    );
    Ok(value.to_vec_i64()?)
}

/// Holds a session's exclusive lease for the length of one pass.
///
/// A lease declared `policy: exclusive` means one writer. Two turns that both
/// read the conversation before either writes it would leave the loser's prompt
/// and generation nowhere, and nothing would report that they were lost. The
/// guard is released on drop, so a pass that fails or panics does not strand the
/// session.
struct SessionLeaseGuard<'a> {
    leases: &'a RefCell<std::collections::HashSet<String>>,
    session: String,
}

impl<'a> SessionLeaseGuard<'a> {
    fn acquire(
        leases: &'a RefCell<std::collections::HashSet<String>>,
        session: &str,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            leases.borrow_mut().insert(session.to_string()),
            "session {session} already has a turn in flight, and its workflow declares the \
             conversation an exclusive lease; run the turns one after another so neither reads a \
             conversation the other is about to replace"
        );
        Ok(Self {
            leases,
            session: session.to_string(),
        })
    }
}

impl Drop for SessionLeaseGuard<'_> {
    fn drop(&mut self) {
        self.leases.borrow_mut().remove(&self.session);
    }
}

/// The SSA value a pass left in a group-backed session cell.
///
/// The group's alias names the `output` port that advances the state; the step
/// that invoked the component names the value that port was bound to. Reading
/// the port through the step is what keeps this independent of how any
/// particular package spells its values.
///
/// The last binding wins: a component invoked several times in one pass has
/// advanced the state each time, and the lease keeps where it ended.
fn session_group_output_value(
    workflow: &WorkflowSpec,
    cell: &str,
    values: &PipelineTensors,
) -> Option<String> {
    fn walk(
        step: &onnx_genai_metadata::WorkflowStep,
        component: &str,
        port: &str,
        values: &PipelineTensors,
        found: &mut Option<String>,
    ) {
        use onnx_genai_metadata::WorkflowStep as Step;
        match step {
            Step::Sequence { steps } => {
                steps
                    .iter()
                    .for_each(|step| walk(step, component, port, values, found));
            }
            Step::Invoke {
                component: invoked,
                outputs,
                ..
            } => {
                if invoked == component
                    && let Some(value) = outputs.get(port)
                    && values.contains_key(value)
                {
                    *found = Some(value.clone());
                }
            }
            Step::Loop { setup, steps, .. } => {
                setup
                    .iter()
                    .chain(steps)
                    .for_each(|step| walk(step, component, port, values, found));
            }
            Step::Branch { cases, default, .. } => {
                cases
                    .values()
                    .for_each(|step| walk(step, component, port, values, found));
                if let Some(default) = default {
                    walk(default, component, port, values, found);
                }
            }
            Step::Emit { .. } => {}
        }
    }
    for (component, alias) in onnx_genai_metadata::session_group_aliases(workflow, cell) {
        let Some(port) = alias.output.as_deref() else {
            continue;
        };
        let mut found = None;
        for step in &workflow.steps {
            walk(step, component, port, values, &mut found);
        }
        if found.is_some() {
            return found;
        }
    }
    None
}

/// Tokens a pass published to one declared output.
///
/// This is the same selection [`super::PipelineOutputs`] makes and that the
/// generate path reads — the aggregate value when the pass published one,
/// otherwise the single request row. A conversation accumulates what a turn
/// generated, and reading it by a second rule would let the session and the
/// caller disagree about what was said.
///
/// A tokens output published as `mode: event` has no aggregate and no row, and
/// the generate path already refuses such a package; there is nothing to
/// accumulate here that a caller ever received.
fn published_conversation_tokens(
    values: &PipelineTensors,
    row_outputs: &BTreeMap<String, Vec<String>>,
    output: &str,
) -> anyhow::Result<Vec<i64>> {
    if let Some(value) = values.get(output) {
        return continuation_tokens(value);
    }
    let rows = row_outputs.get(output).map(Vec::as_slice).unwrap_or(&[]);
    anyhow::ensure!(
        rows.len() <= 1,
        "a session continues one conversation, but this pass published {} request rows to \
         '{output}'",
        rows.len()
    );
    match rows.first().and_then(|name| values.get(name)) {
        Some(value) => continuation_tokens(value),
        None => Ok(Vec::new()),
    }
}

/// Whether this workflow declares state a session can actually carry.
///
/// Answered by [`onnx_genai_metadata::classify_session_state`], which is also
/// what the validator reads, so a package the document blesses and a package
/// this runtime will open a session for are the same set. Computing it twice is
/// how they came to disagree about state service groups.
pub fn workflow_carries_session_state(workflow: &WorkflowSpec) -> bool {
    onnx_genai_metadata::classify_session_state(workflow).carries_any()
}

impl WorkflowRuntime {
    fn materialize_workflow_value_copy(&self, value: &Value) -> anyhow::Result<Value> {
        if value.is_host_resident()? {
            return clone_value(value);
        }
        // Every device→host copy this runtime performs goes through here, which
        // is why the count lives here and not at the call sites: a new caller
        // is counted without having to remember to count itself.
        self.host_staging_count
            .set(self.host_staging_count.get().saturating_add(1));
        let device_id = value.device_id()?;
        if let Some(island) = self
            .execution_islands
            .iter()
            .find(|island| island.cuda_device_id() == Some(device_id))
        {
            return island.materialize_host(value);
        }
        // Native backend: no execution islands exist, but a native component may
        // still hand back a device-resident value. Materialize it to host with a
        // standalone device→host copy (the interpreter's host-policy boundaries
        // — branch predicates, adapters, emit — need host bytes).
        value
            .to_host_from_cuda(device_id)
            .with_context(|| format!("failed to materialize CUDA device {device_id} value to host"))
    }

    fn materialize_workflow_value(
        &self,
        values: &mut PipelineTensors,
        name: &str,
    ) -> anyhow::Result<()> {
        let value = values
            .get(name)
            .with_context(|| format!("workflow value '{name}' is unavailable"))?;
        if value.is_host_resident()? {
            return Ok(());
        }
        let device_id = value.device_id()?;
        let host = if let Some(island) = self
            .execution_islands
            .iter()
            .find(|island| island.cuda_device_id() == Some(device_id))
        {
            island.materialize_host(value)?
        } else {
            // Native backend: no islands; standalone device→host copy.
            value.to_host_from_cuda(device_id).with_context(|| {
                format!("failed to materialize CUDA device {device_id} value '{name}' to host")
            })?
        };
        values.insert(name.to_string(), host);
        Ok(())
    }

    pub(crate) fn package_outputs(
        &self,
        mut values: PipelineTensors,
        row_outputs: BTreeMap<String, Vec<String>>,
    ) -> anyhow::Result<PipelineOutputs> {
        let mut outputs = PipelineTensors::new();
        for output in self.workflow.outputs.keys() {
            let row_prefix = format!("{output}.row.");
            let event_prefix = format!("{output}.");
            let names = values
                .keys()
                .filter(|name| {
                    *name == output
                        || name.starts_with(&row_prefix)
                        || (name.starts_with(&event_prefix)
                            && name[event_prefix.len()..]
                                .chars()
                                .all(|character| character.is_ascii_digit()))
                })
                .cloned()
                .collect::<Vec<_>>();
            for name in names {
                self.materialize_workflow_value(&mut values, &name)?;
                outputs.insert(
                    name.clone(),
                    values
                        .remove(&name)
                        .with_context(|| format!("workflow package output '{name}' disappeared"))?,
                );
            }
        }
        Ok(PipelineOutputs {
            tensors: outputs,
            rows: row_outputs,
        })
    }
    /// Why the last generation stopped.
    ///
    /// `true` when the outermost loop's liveness predicate went false — the
    /// model's own stop — and `false` when it exhausted its iteration bound.
    /// Read by the generate path so a composite workflow reports the reason it
    /// actually finished for.
    pub(crate) fn last_generation_ended_by_predicate(&self) -> bool {
        self.workflow_performance
            .borrow()
            .last_loop_ended_by_predicate
    }

    pub fn workflow_performance_diagnostic(&self) -> WorkflowPerformanceDiagnostic {
        let counters = self.workflow_performance.borrow();
        let elapsed_seconds = counters.last_elapsed_ns as f64 / 1_000_000_000.0;
        WorkflowPerformanceDiagnostic {
            runs: counters.runs,
            total_elapsed_ns: counters.total_elapsed_ns,
            last_elapsed_ns: counters.last_elapsed_ns,
            last_ttft_ns: counters.last_ttft_ns,
            last_emit_timestamps_ns: counters.last_emit_timestamps_ns.clone(),
            last_loop_iterations: counters.last_loop_iterations,
            last_component_invocations: counters.last_component_invocations,
            last_stage_runs: counters.last_stage_runs.clone(),
            last_stage_elapsed_ns: counters.last_stage_elapsed_ns.clone(),
            last_emit_events: counters.last_emit_events,
            last_emitted_elements: counters.last_emitted_elements,
            last_steps_per_second: if elapsed_seconds > 0.0 {
                counters.last_loop_iterations as f64 / elapsed_seconds
            } else {
                0.0
            },
            last_elements_per_second: if elapsed_seconds > 0.0 {
                counters.last_emitted_elements as f64 / elapsed_seconds
            } else {
                0.0
            },
            islands: self.execution_island_diagnostics(),
        }
    }

    pub(crate) fn run_workflow(
        &self,
        request: PipelineGenerateRequest,
    ) -> anyhow::Result<PipelineTensors> {
        WorkflowExecutionPlan::new(self, request)?.execute()
    }

    pub(crate) fn run_workflow_outputs(
        &self,
        request: PipelineGenerateRequest,
    ) -> anyhow::Result<PipelineOutputs> {
        WorkflowExecutionPlan::new(self, request)?.execute_outputs()
    }

    pub(crate) fn run_workflow_retained(
        &self,
        request: PipelineGenerateRequest,
    ) -> anyhow::Result<PipelineTensors> {
        let mut plan = WorkflowExecutionPlan::new(self, request)?;
        let (mut values, _) = plan.execute_retained()?;
        // `execute_retained` parks the pass's inputs back on the plan for replay.
        // A retained run has no replay — the caller wanted the whole value map —
        // so fold them back in; a proposal chain binds the proposer's borrowed
        // shared KV and masks straight from these request values.
        for (name, value) in std::mem::take(&mut plan.values) {
            values.entry(name).or_insert(value);
        }
        // Declared package outputs read the same here as they do from
        // `run_pipeline`: host-resident. Internal SSA values are deliberately
        // left where the backend produced them, so a native-CUDA proposal chain
        // keeps its carry and borrowed KV device-resident instead of paying a
        // host round-trip to be observed.
        for output in self.workflow.outputs.keys() {
            let row_prefix = format!("{output}.row.");
            let event_prefix = format!("{output}.");
            let names = values
                .keys()
                .filter(|name| {
                    *name == output
                        || name.starts_with(&row_prefix)
                        || (name.starts_with(&event_prefix)
                            && name[event_prefix.len()..]
                                .chars()
                                .all(|character| character.is_ascii_digit()))
                })
                .cloned()
                .collect::<Vec<_>>();
            for name in names {
                self.materialize_workflow_value(&mut values, &name)?;
            }
        }
        Ok(values)
    }
}

impl WorkflowRuntime {
    /// Rejoin a session's conversation to this turn's prompt.
    ///
    /// Returns the request the pass actually runs and, when a continuation
    /// applies, the token row the prompt input now carries. A request with no
    /// session, or a package declaring no continuation, is returned untouched —
    /// stateless generation is the same execution it always was, and this is
    /// where that is guaranteed rather than in each caller.
    fn continue_session_prompt(
        &self,
        request: GenerateRequest,
        session_id: Option<&str>,
    ) -> anyhow::Result<(GenerateRequest, Option<Vec<i64>>)> {
        let Some((cell, _, _, _)) = workflow_prompt_continuation(&self.workflow) else {
            return Ok((request, None));
        };
        let Some(session_id) = session_id else {
            return Ok((request, None));
        };
        let history = match self
            .workflow_session_state
            .borrow()
            .get(&(session_id.to_string(), cell.to_string()))
        {
            Some(value) => continuation_tokens(value)
                .with_context(|| format!("session state '{cell}' is not a conversation"))?,
            None => Vec::new(),
        };
        let turn: Vec<i64> = match &request.prompt {
            GeneratePrompt::TokenIds(tokens) => {
                tokens.iter().map(|token| i64::from(*token)).collect()
            }
            GeneratePrompt::TokenRows(rows) => {
                anyhow::ensure!(
                    rows.len() == 1,
                    "a session continues one conversation, but this request carries {} prompt \
                     rows; batch several requests instead of several rows in a session",
                    rows.len()
                );
                rows[0].iter().map(|token| i64::from(*token)).collect()
            }
            GeneratePrompt::Text(_) => anyhow::bail!(
                "a session-continued prompt must reach the workflow as token ids; this request \
                 still carries text"
            ),
        };
        let mut conversation: Vec<i64> = history;
        conversation.extend_from_slice(&turn);
        let tokens = conversation
            .iter()
            .map(|token| u32::try_from(*token))
            .collect::<Result<Vec<_>, _>>()
            .context("a conversation token id does not fit the prompt token type")?;
        Ok((
            GenerateRequest {
                prompt: GeneratePrompt::TokenIds(tokens),
                options: request.options,
            },
            Some(conversation),
        ))
    }
}

impl<'a> WorkflowExecutionPlan<'a> {
    pub(crate) fn new(
        engine: &'a WorkflowRuntime,
        request: PipelineGenerateRequest,
    ) -> anyhow::Result<Self> {
        Self::new_hosted(engine, request, &[])
    }

    /// Prepare a pass in which `hosted` contracts are executed by the caller.
    ///
    /// The contract list is the caller's statement about what it can run, and
    /// it is the only thing that makes a declared input optional here: an input
    /// read solely by a step the caller executes is that executor's business,
    /// and demanding it of the request would be asking for a second answer to a
    /// question the executor has already answered. An input any generically
    /// executed component reads stays required.
    pub(crate) fn new_hosted(
        engine: &'a WorkflowRuntime,
        request: PipelineGenerateRequest,
        hosted: &[&str],
    ) -> anyhow::Result<Self> {
        let PipelineGenerateRequest {
            request,
            inputs,
            session_id,
            component_overrides,
        } = request;
        let workflow = &engine.workflow;
        validate_component_overrides(workflow, &component_overrides)?;
        // A conversation the package declares as a prompt prefix rejoins here,
        // before anything is bound: every input derived from the prompt — the
        // token tensor, its length, the mask the initializer builds from it —
        // is then derived from the *whole* conversation, exactly as it would be
        // for a caller who sent it in one request. Rewriting the bound tensors
        // afterwards would leave the derived ones describing only this turn.
        let (request, continued_prompt) =
            engine.continue_session_prompt(request, session_id.as_deref())?;
        let host_supplied = if hosted.is_empty() {
            std::collections::HashSet::new()
        } else {
            super::generation::host_supplied_inputs(workflow, hosted)
        };
        let runtime_managed_seeds = super::generation::runtime_managed_seeds(workflow);
        let serving_controls = super::generation::serving_control_seeds(workflow);
        let (mut values, from_literal) = engine.bind_workflow_inputs(
            workflow,
            &request,
            inputs,
            &host_supplied,
            &runtime_managed_seeds,
            &serving_controls,
        )?;
        let dynamic_symbols = workflow
            .state
            .values()
            .filter_map(|state| match &state.recurrence {
                onnx_genai_metadata::ShapeRecurrence::Growing { axis, .. }
                | onnx_genai_metadata::ShapeRecurrence::Bounded { axis, .. } => state
                    .contract
                    .shape
                    .as_ref()
                    .and_then(|shape| shape.get(*axis))
                    .and_then(|dimension| match dimension {
                        TensorDimension::Symbol(symbol) => Some(symbol.clone()),
                        TensorDimension::Fixed(_) => None,
                    }),
                onnx_genai_metadata::ShapeRecurrence::Invariant => None,
            })
            .collect::<std::collections::HashSet<_>>();
        let mut initial_symbols = HashMap::new();
        // Two passes: caller/request-supplied values bind the workflow's symbolic
        // axes first, then literal-materialized defaults are re-shaped to the
        // extents those bindings imply. `literal_shape` can only guess a
        // singleton for a symbolic axis, so binding literals first would pin a
        // shared symbol such as `batch` to 1 and reject every batched request.
        for (name, input) in &workflow.inputs {
            if from_literal.contains_key(name) {
                continue;
            }
            if let Some(value) = values.get(name) {
                validate_workflow_value(
                    name,
                    value,
                    &input.contract,
                    &mut initial_symbols,
                    &dynamic_symbols,
                )?;
            }
        }
        for (name, input) in &workflow.inputs {
            let Some(literal) = from_literal.get(name) else {
                continue;
            };
            let Some(value) = values.get(name) else {
                continue;
            };
            let resolved = resolve_workflow_shape(&input.contract, &initial_symbols)
                .unwrap_or_else(|_| value.shape().to_vec());
            if resolved != value.shape() {
                let rebuilt =
                    workflow_literal_value_with_shape(literal, &input.contract, &resolved)?;
                values.insert(name.clone(), rebuilt);
            }
            let value = values
                .get(name)
                .expect("literal workflow input was just materialized");
            validate_workflow_value(
                name,
                value,
                &input.contract,
                &mut initial_symbols,
                &dynamic_symbols,
            )?;
        }
        let input_names = values.keys().cloned().collect::<Vec<_>>();
        // The bound the continuation cell declares is the package's own context
        // limit, and this is the only place it can be checked: a continuation is
        // deliberately not loop-carried, so it never reaches the carry path's
        // recurrence check.
        //
        // It is checked against the conversation *and the budget this turn was
        // admitted with*, because a turn refused after it has generated is a
        // turn whose work and whose stream are thrown away. Admission is the
        // last moment at which refusing costs nothing.
        if let Some(conversation) = continued_prompt.as_deref()
            && let Some((cell, _, _, state)) = workflow_prompt_continuation(workflow)
        {
            let bound = continuation_bound(cell, state, &values)?;
            let budget = request.options.max_new_tokens;
            anyhow::ensure!(
                conversation.len().saturating_add(budget) <= bound,
                "{CONVERSATION_OVER_BOUND}: this turn would leave {} tokens ({} of conversation \
                 and up to {budget} generated) and '{cell}' declares a bound of {bound}; reset or \
                 close the session to start a new one, or ask for fewer tokens",
                conversation.len() + budget,
                conversation.len()
            );
        }
        let input_aliases = workflow
            .inputs
            .iter()
            .filter_map(|(name, input)| match &input.source {
                WorkflowInputSource::Application { name: alias } => {
                    Some((alias.clone(), name.clone()))
                }
                _ => None,
            })
            .chain(
                workflow
                    .inputs
                    .keys()
                    .map(|name| (name.clone(), name.clone())),
            )
            .collect();
        Ok(Self {
            engine,
            values,
            input_names,
            input_aliases,
            initial_symbols,
            dynamic_symbols,
            session_id,
            component_overrides,
            max_iterations_only: !request.options.stop_on_eos,
            continued_prompt,
        })
    }

    /// Replace a prepared package/application input without rebuilding the plan.
    pub fn set_input(&mut self, name: &str, value: Value) -> anyhow::Result<()> {
        let package_name = self
            .input_aliases
            .get(name)
            .with_context(|| format!("workflow execution plan has no input '{name}'"))?;
        let input = self
            .engine
            .workflow
            .inputs
            .get(package_name)
            .with_context(|| format!("workflow package input '{package_name}' is undeclared"))?;
        let mut symbols = self.initial_symbols.clone();
        validate_workflow_value(
            package_name,
            &value,
            &input.contract,
            &mut symbols,
            &self.dynamic_symbols,
        )?;
        self.values.insert(package_name.clone(), value);
        if !self.input_names.contains(package_name) {
            self.input_names.push(package_name.clone());
        }
        Ok(())
    }

    /// Execute the already-bound workflow and retain its input slots for replay.
    pub fn execute(&mut self) -> anyhow::Result<PipelineTensors> {
        self.execute_outputs().map(PipelineOutputs::into_tensors)
    }

    pub fn execute_outputs(&mut self) -> anyhow::Result<PipelineOutputs> {
        let (values, row_outputs) = self.execute_retained()?;
        self.engine.package_outputs(values, row_outputs)
    }

    /// Execute the workflow and keep every SSA value the pass produced.
    ///
    /// [`Self::execute_outputs`] narrows this to the package's declared outputs.
    /// Interpreter-level constructs that consume a completed pass — the chained
    /// speculative driver reading its `folded_carry_seed` and the proposer's
    /// borrowed shared-KV bindings, above all — need the values the workflow
    /// bound internally, not just what it published.
    pub fn execute_retained(
        &mut self,
    ) -> anyhow::Result<(PipelineTensors, BTreeMap<String, Vec<String>>)> {
        self.execute_retained_with_host(&mut None)
    }

    /// Execute with a host supplying the runtime-implemented nodes.
    ///
    /// A workflow may declare steps the package ships no graph for — an
    /// autoregressive decoder driven by a fused session, or the token policy.
    /// The host implements those; the interpreter still owns the loop, the
    /// carries and the emits, so there is one execution machinery rather than a
    /// second loop beside it.
    pub(crate) fn execute_retained_with_host(
        &mut self,
        host: &mut Option<&mut dyn WorkflowNodeHost>,
    ) -> anyhow::Result<(PipelineTensors, BTreeMap<String, Vec<String>>)> {
        let started = std::time::Instant::now();
        let mut telemetry = WorkflowRunTelemetry {
            started: Some(started),
            max_iterations_only: self.max_iterations_only,
            ..WorkflowRunTelemetry::default()
        };
        let engine = self.engine;
        let generation = engine.workflow_execution_generation.get().wrapping_add(1);
        engine.workflow_execution_generation.set(generation);
        for island in &engine.execution_islands {
            island.begin_execution(generation);
        }
        let workflow = &engine.workflow;
        let mut values = std::mem::take(&mut self.values);
        let prepare_adapters = (|| -> anyhow::Result<()> {
            if let Some(service) = &engine.adapter_service {
                if !service.portable_fallback {
                    anyhow::bail!(
                        "adapter capability '{}' is unavailable in the portable workflow runtime and portable_fallback is disabled",
                        service.application_capability
                    );
                }
                // Adapter rows are positional: row i of the request-aligned
                // selection tensors belongs to batch row i. The runtime, not
                // the package, associates a batch row with a request.
                let adapter_counts = values
                    .get(&service.selection.adapter_counts)
                    .with_context(|| {
                        format!(
                            "adapter service counts input '{}' is unavailable",
                            service.selection.adapter_counts
                        )
                    })?
                    .to_vec_i64()
                    .with_context(|| {
                        format!(
                            "adapter service counts input '{}' must be host int64",
                            service.selection.adapter_counts
                        )
                    })?;
                let batch_rows = adapter_counts.len();
                let active_rows = if let Some(active) = &service.selection.active {
                    workflow_bool_rows(&values, active)?
                } else {
                    vec![true; batch_rows]
                };
                let selection =
                    super::adapters::selection_from_inputs(service, &values, batch_rows)?;
                let context = engine.adapter_cache.borrow_mut().prepare(
                    &engine.package_root,
                    service,
                    &selection,
                    &active_rows,
                )?;
                *engine.active_adapter_context.borrow_mut() = Some(context);
                // A runtime-minted row selection expands or compacts the batch
                // (beam search, speculative row cloning). It carries source
                // positions within the current batch, never scheduler IDs, so
                // every row-scoped component follows through the same ABI.
                if let Some(selection) = workflow_row_selection(workflow, &values)?
                    && let Some(context) = engine.active_adapter_context.borrow_mut().as_mut()
                {
                    context.compact(&selection)?;
                }
            }
            Ok(())
        })();
        if let Err(error) = prepare_adapters {
            self.retain_inputs(&mut values);
            return Err(error);
        }
        let _adapter_context_guard = ActiveAdapterContextGuard(&engine.active_adapter_context);
        // A package may declare session state and still be asked for a single
        // stateless generation — that is the first turn of every conversation,
        // and it is also what `Engine::generate` is. Without a session there is
        // no lease to read, so each cell starts from the initializer it
        // declares; refusing here would make declaring a conversation cost a
        // package the ability to answer one question.
        let session_state = onnx_genai_metadata::classify_session_state(workflow);
        // Taken before the lease is read and released when the pass ends,
        // whichever way it ends.
        let _session_lease = match (self.session_id.as_deref(), session_state.carries_any()) {
            (Some(session_id), true) => {
                match SessionLeaseGuard::acquire(&engine.workflow_session_leases, session_id) {
                    Ok(guard) => Some(guard),
                    Err(error) => {
                        self.retain_inputs(&mut values);
                        return Err(error);
                    }
                }
            }
            _ => None,
        };
        if let Some(session_id) = self.session_id.as_ref() {
            for (cell, carrier) in session_state.carried() {
                let Some(value) = engine
                    .workflow_session_state
                    .borrow()
                    .get(&(session_id.to_string(), cell.to_string()))
                    .map(clone_value)
                    .transpose()?
                else {
                    continue;
                };
                match carrier {
                    // A loop reads its lease where it seeds the carry.
                    onnx_genai_metadata::SessionStateCarrier::LoopCarry => {
                        values.insert(session_state_value_name(cell), value);
                    }
                    // A group holds the storage, and the graph reaches it
                    // through the value the cell's initializer names — which is
                    // the port the group's alias declares. Replacing it for the
                    // pass is what makes the lease the `past` the component
                    // reads, rather than the seed the caller sent.
                    onnx_genai_metadata::SessionStateCarrier::StateServiceGroup => {
                        let initializer = &workflow
                            .state
                            .get(cell)
                            .expect("a classified cell is declared")
                            .initializer;
                        values.insert(initializer.clone(), value);
                    }
                    // Bound into the prompt before this plan was built.
                    onnx_genai_metadata::SessionStateCarrier::PromptContinuation => {}
                }
            }
        }
        let mut symbols = self.initial_symbols.clone();
        let mut emit_counts = HashMap::new();
        let mut final_state_refs = HashMap::new();
        let result = engine.run_workflow_node(
            &engine.compiled_workflow.graph,
            workflow,
            &mut values,
            &mut symbols,
            &self.dynamic_symbols,
            &mut emit_counts,
            &mut final_state_refs,
            &self.component_overrides,
            &mut telemetry,
            host,
        );
        if let Err(error) = result {
            self.retain_inputs(&mut values);
            return Err(error);
        }
        for output in workflow_emitted_outputs(&engine.compiled_workflow.graph) {
            let Some(value) = values.get(&output) else {
                continue;
            };
            let contract = &workflow
                .outputs
                .get(&output)
                .with_context(|| format!("workflow emitted undeclared output '{output}'"))?
                .contract;
            validate_workflow_value(
                &output,
                value,
                contract,
                &mut symbols,
                &self.dynamic_symbols,
            )?;
        }
        if let Some(session_id) = &self.session_id {
            let continuation = workflow_prompt_continuation(workflow)
                .map(|(cell, _, tokens_output, _)| (cell.to_string(), tokens_output.to_string()));
            let mut updates = Vec::new();
            for (cell, carrier) in session_state.carried() {
                let state = workflow
                    .state
                    .get(cell)
                    .expect("a classified cell is declared");
                // The conversation this pass leaves behind is the prompt it ran
                // — which already carried every earlier turn — followed by what
                // it published. Reading it off the emitted stream alone would
                // drop everything before this turn; reading it off the cell's
                // final SSA value would drop the turn, because no step in the
                // workflow writes the conversation.
                if let Some((continued_cell, tokens_output)) = &continuation
                    && continued_cell == cell
                {
                    let prompt = self.continued_prompt.as_deref().unwrap_or_default();
                    let mut conversation = prompt.to_vec();
                    conversation.extend(published_conversation_tokens(
                        &values,
                        &telemetry.row_outputs,
                        tokens_output,
                    )?);
                    let length = conversation.len() as i64;
                    // Admission already refused a turn whose conversation plus
                    // budget exceeds the bound, so reaching this means the pass
                    // published more tokens than it was admitted for. Storing it
                    // would leave the session in a state its own declaration
                    // forbids and every later turn would fail the same way with
                    // no way back, so this refuses — and names the package,
                    // because a caller cannot cause it.
                    let bound = continuation_bound(cell, state, &values)?;
                    if conversation.len() > bound {
                        anyhow::bail!(
                            "{CONVERSATION_OVER_BOUND}: this package published more tokens than \
                             the turn was admitted for, leaving a conversation of {} tokens \
                             against a declared bound of {bound} on '{cell}'",
                            conversation.len()
                        );
                    }
                    let shape = match state.contract.rank {
                        1 => vec![length],
                        2 => vec![1, length],
                        rank => anyhow::bail!(
                            "session continuation state '{cell}' must have rank 1 or 2, got {rank}"
                        ),
                    };
                    updates.push((
                        (session_id.clone(), cell.to_string()),
                        Value::from_slice_i64(&conversation, &shape)?,
                    ));
                    continue;
                }
                // What the next lease holds is what this pass left in the
                // cell. A loop carry records that in `final_state_refs`. A
                // group-backed cell is advanced by the graph writing the
                // alias's `output` port, so the value bound to that port is the
                // one to keep — reading the initializer instead would store the
                // value the pass *started* from and replay it forever.
                let group_output = (carrier
                    == onnx_genai_metadata::SessionStateCarrier::StateServiceGroup)
                    .then(|| session_group_output_value(workflow, cell, &values))
                    .flatten();
                let value_ref = group_output.as_deref().unwrap_or_else(|| {
                    final_state_refs
                        .get(cell)
                        .map(String::as_str)
                        .unwrap_or(&state.initializer)
                });
                let value = values.get(value_ref).with_context(|| {
                    format!(
                        "session-scoped workflow state '{cell}' has no final value '{value_ref}'"
                    )
                })?;
                updates.push(((session_id.clone(), cell.to_string()), clone_value(value)?));
            }
            let mut session_state = engine.workflow_session_state.borrow_mut();
            for (key, value) in updates {
                session_state.insert(key, value);
            }
        }
        let row_outputs = std::mem::take(&mut telemetry.row_outputs);
        engine.publish_workflow_telemetry(telemetry);
        let inputs = self.take_inputs(&mut values);
        self.values = inputs;
        Ok((values, row_outputs))
    }

    fn retain_inputs(&mut self, values: &mut PipelineTensors) {
        self.values = self.take_inputs(values);
    }

    fn take_inputs(&self, values: &mut PipelineTensors) -> PipelineTensors {
        self.input_names
            .iter()
            .filter_map(|name| values.remove(name).map(|value| (name.clone(), value)))
            .collect()
    }
}

impl WorkflowRuntime {
    /// Fold one pass's telemetry into this runtime's performance counters.
    ///
    /// One place, so a pass driven a token at a time reports the same counters
    /// as one driven to completion — including `loop_ended_by_predicate`, which
    /// is how a caller learns whether generation stopped at its own EOS or ran
    /// out of budget.
    fn publish_workflow_telemetry(&self, telemetry: WorkflowRunTelemetry) {
        let elapsed_ns = telemetry
            .started
            .map_or(0, |started| started.elapsed().as_nanos());
        let mut counters = self.workflow_performance.borrow_mut();
        counters.runs += 1;
        counters.total_elapsed_ns += elapsed_ns;
        counters.last_elapsed_ns = elapsed_ns;
        counters.last_ttft_ns = telemetry.first_emit_ns;
        counters.last_emit_timestamps_ns = telemetry.emit_timestamps_ns;
        counters.last_loop_iterations = telemetry.loop_iterations;
        counters.last_loop_ended_by_predicate = telemetry.loop_ended_by_predicate;
        counters.last_component_invocations = telemetry.component_invocations;
        counters.last_stage_runs = telemetry.stage_runs;
        counters.last_stage_elapsed_ns = telemetry.stage_elapsed_ns;
        counters.last_emit_events = telemetry.emit_events;
        counters.last_emitted_elements = telemetry.emitted_elements;
    }
}

/// Contracts this runtime supplies an executor for.
///
/// Mirrors [`workflow_adapter_registry`]: a workflow names a contract, and the
/// runtime either has an executor for it or the package declares something this
/// build cannot run. Keeping the set in one place is what makes "which steps can
/// this runtime implement?" a question with a single answer, rather than a
/// property of whichever `match` arm a reader happens to find.
///
/// Membership here is *necessary but not sufficient*: a host still declines a
/// contract it holds no resources for (no decode session, say), and the
/// interpreter reports that as an unimplemented contract rather than silently
/// continuing.
pub(crate) fn runtime_contract_registry() -> &'static std::collections::HashSet<&'static str> {
    static REGISTRY: std::sync::LazyLock<std::collections::HashSet<&'static str>> =
        std::sync::LazyLock::new(|| {
            std::collections::HashSet::from([
                onnx_genai_metadata::decoder_workflow::AUTOREGRESSIVE_DECODE_CONTRACT,
                onnx_genai_metadata::decoder_workflow::TOKEN_POLICY_CONTRACT,
                onnx_genai_metadata::decoder_workflow::CONTINUOUS_BATCH_CONTRACT,
                onnx_genai_metadata::decoder_workflow::SPECULATIVE_BLOCK_CONTRACT,
            ])
        });
    &REGISTRY
}

/// A runtime-supplied executor for a workflow node the package does not ship a
/// graph for.
///
/// A workflow declares *what happens* at each step. Most steps are ONNX graphs
/// the package ships; some are steps the runtime implements — an autoregressive
/// decoder driven by a fused session that owns paged KV on device, or the token
/// policy that scores logits and decides when generation ends. A package states
/// those as `implementation: binding` with a contract id, and the runtime
/// supplies the implementation here.
///
/// This is the seam that lets the interpreter own the loop while a backend owns
/// the step. The interpreter still resolves `continue_when`, `max_iterations`,
/// `carried`, `termination` and the emits; the host only advances one node. A
/// host that owned the loop instead would be a second runtime, which is exactly
/// what this replaces.
pub(crate) trait WorkflowNodeHost {
    /// Contracts this host implements.
    ///
    /// Declared rather than discovered so the interpreter can hand a node to
    /// the host *before* resolving the node's inputs: an executor that supplies
    /// its own mask and positions must not have them demanded of the caller
    /// first. It is also what the execution plan reads to decide which declared
    /// package inputs are the host's business.
    fn hosted_contracts(&self) -> &'static [&'static str];

    /// Execute one node bound to `contract`.
    ///
    /// Returns `Ok(false)` when this host does not implement the contract, so
    /// the interpreter can report an unimplemented contract rather than
    /// silently doing nothing — a declared step that no one runs is the failure
    /// mode this signature exists to prevent.
    fn execute_contract_node(&mut self, request: WorkflowNodeRequest<'_>) -> anyhow::Result<bool>;
}

/// Everything a host needs to advance one declared node.
///
/// `dead_code` is allowed only until the engine-side executors land: this
/// checkpoint defines the seam and the interpreter's dispatch, and the first
/// non-test implementor is the autoregressive-decode host in the next commit.
/// The allowance is scoped to this struct so it cannot hide anything else.
#[allow(dead_code)]
pub(crate) struct WorkflowNodeRequest<'a> {
    /// Contract id the component declares.
    pub(crate) contract: &'a str,
    /// Component name, for diagnostics only — never for dispatch.
    pub(crate) component: &'a str,
    /// Port name to SSA value, as the invocation bound them.
    pub(crate) inputs: &'a BTreeMap<String, String>,
    /// Port name to SSA value the host must define.
    pub(crate) outputs: &'a BTreeMap<String, String>,
    /// The invocation's value environment.
    pub(crate) values: &'a mut PipelineTensors,
}

impl WorkflowRuntime {
    // Recursive execution threads the explicit interpreter stores and telemetry.
    #[allow(clippy::too_many_arguments)]
    fn run_workflow_node(
        &self,
        node: &WorkflowNode,
        workflow: &WorkflowSpec,
        values: &mut PipelineTensors,
        symbols: &mut HashMap<String, i64>,
        dynamic_symbols: &std::collections::HashSet<String>,
        emit_counts: &mut HashMap<String, usize>,
        final_state_refs: &mut HashMap<String, String>,
        component_overrides: &HashMap<String, String>,
        telemetry: &mut WorkflowRunTelemetry,
        host: &mut Option<&mut dyn WorkflowNodeHost>,
    ) -> anyhow::Result<()> {
        match node {
            WorkflowNode::Sequence { nodes } => {
                for node in nodes {
                    self.run_workflow_node(
                        node,
                        workflow,
                        values,
                        symbols,
                        dynamic_symbols,
                        emit_counts,
                        final_state_refs,
                        component_overrides,
                        telemetry,
                        host,
                    )?;
                }
            }
            WorkflowNode::Invoke {
                component,
                inputs,
                outputs,
                ..
            } => {
                let stage_started = std::time::Instant::now();
                telemetry.component_invocations += 1;
                let declaration = workflow
                    .components
                    .get(component)
                    .with_context(|| format!("workflow component '{component}' is undeclared"))?;
                match &declaration.implementation {
                    ComponentImplementation::Onnx { .. } => {
                        // A component may declare *what step it is* alongside
                        // the graph that implements it. When this runtime holds
                        // a registered executor for that contract — a fused
                        // decode session owning paged KV and a captured graph,
                        // above all — it runs the declared step instead of
                        // invoking the artifact generically. Both run the same
                        // node; which executor does is answered by the
                        // contract, never by inspecting the package.
                        //
                        // Consulted before the inputs are resolved on purpose:
                        // such an executor supplies the ports it owns (the
                        // attention mask it builds from the sequence it holds),
                        // and demanding them of the caller first would ask for
                        // a second answer to a question it has already
                        // answered.
                        if let Some(contract) = declaration
                            .contract
                            .as_ref()
                            .map(|contract| contract.id.as_str())
                            && let Some(host) = host.as_deref_mut()
                            && host.hosted_contracts().contains(&contract)
                        {
                            let handled = host.execute_contract_node(WorkflowNodeRequest {
                                contract,
                                component,
                                inputs,
                                outputs,
                                values,
                            })?;
                            anyhow::ensure!(
                                handled,
                                "workflow component '{component}' declares contract \
                                 '{contract}', which the running host lists as implemented but \
                                 then declined to execute"
                            );
                            self.record_contract_execution(contract);
                            telemetry.record_stage(
                                component.clone(),
                                stage_started.elapsed().as_nanos(),
                            );
                            return Ok(());
                        }
                        let (
                            selected_component,
                            selected_declaration,
                            selected_inputs,
                            selected_outputs,
                        ) = resolve_component_invocation(
                            workflow,
                            component,
                            declaration,
                            inputs,
                            outputs,
                            component_overrides,
                        )?;
                        // Fixed-capacity groups scatter to positions carried in
                        // data. Bounds-check them here, while an illegal write
                        // is still a diagnosable error rather than corruption.
                        self.enforce_fixed_capacity_writes(
                            workflow,
                            selected_component,
                            &selected_inputs,
                            values,
                        )?;
                        // Component dimensions are invocation-local. A decoder, for example, may
                        // bind `sequence` to the prompt length in setup and to one in the loop.
                        // Values crossing the package boundary were already checked there; the
                        // component contract unifies its own ports without conflating equal
                        // spelling in separate contract scopes.
                        let mut component_symbols = HashMap::new();
                        let component_dynamic_symbols = std::collections::HashSet::new();
                        let resolved = selected_inputs
                            .iter()
                            .map(|(port, value)| {
                                values
                                    .get(value)
                                    .with_context(|| {
                                        format!(
                                            "workflow component '{selected_component}' input '{port}' \
                                             references unavailable value '{value}'"
                                        )
                                    })
                                    .and_then(|tensor| {
                                        if let Some(contract) =
                                            selected_declaration.ports.inputs.get(port)
                                        {
                                            validate_workflow_value(
                                                value,
                                                tensor,
                                                contract,
                                                &mut component_symbols,
                                                &component_dynamic_symbols,
                                            )?;
                                        }
                                        Ok((port.as_str(), tensor))
                                    })
                            })
                            .collect::<anyhow::Result<Vec<_>>>()?;
                        // Concrete output shapes known from the bound input
                        // symbols, so the native CUDA seam can keep resolvable
                        // outputs (a recurring/state tensor in particular)
                        // device-resident. The ORT path ignores this.
                        let component_output_shapes = resolve_component_output_shapes(
                            selected_declaration,
                            &selected_outputs,
                            &component_symbols,
                            &component_dynamic_symbols,
                        );
                        let produced = self.invoke_onnx_component(
                            workflow,
                            component,
                            selected_component,
                            selected_declaration,
                            &component_symbols,
                            &resolved,
                            &selected_outputs,
                            &component_output_shapes,
                        )?;
                        for (port, tensor) in produced {
                            let Some(value) = selected_outputs.get(&port) else {
                                continue;
                            };
                            if let Some(contract) = selected_declaration.ports.outputs.get(&port) {
                                validate_workflow_value(
                                    value,
                                    &tensor,
                                    contract,
                                    &mut component_symbols,
                                    &component_dynamic_symbols,
                                )?;
                            }

                            values.insert(value.clone(), tensor);
                        }
                    }
                    ComponentImplementation::Binding => {
                        // A binding says "a step happens here and the runtime
                        // implements it". When it names a contract, the host is
                        // the implementation; the interpreter still owns the
                        // loop around it.
                        let contract = declaration.contract.as_ref().map(|c| c.id.as_str());
                        let handled = match (contract, host.as_deref_mut()) {
                            (Some(contract), Some(host)) => {
                                let handled = host.execute_contract_node(WorkflowNodeRequest {
                                    contract,
                                    component,
                                    inputs,
                                    outputs,
                                    values,
                                })?;
                                if handled {
                                    self.record_contract_execution(contract);
                                }
                                handled
                            }
                            _ => false,
                        };
                        if !handled {
                            // A contract nobody implements must fail, not fall
                            // through to a port copy that would define the
                            // outputs with the wrong tensors and let a declared
                            // step silently not happen.
                            if let Some(contract) = contract {
                                // Two different problems, and an operator can act
                                // on only one of them: a contract this build has
                                // no executor for is a package/runtime version
                                // mismatch, while a registered contract that went
                                // unhandled means the host reached this step
                                // without the resources to run it.
                                if runtime_contract_registry().contains(contract) {
                                    anyhow::bail!(
                                        "workflow component '{component}' declares contract \
                                         '{contract}', which this runtime implements but no \
                                         executor was available for this request"
                                    );
                                }
                                anyhow::bail!(
                                    "workflow component '{component}' declares contract \
                                     '{contract}', which no runtime executor implements"
                                );
                            }
                            for (port, output) in outputs {
                                let source = inputs.get(port).with_context(|| {
                                    format!(
                                        "binding component '{component}' output '{port}' requires \
                                         an input with the same port name"
                                    )
                                })?;
                                let tensor = values.get(source).with_context(|| {
                                    format!("binding source value '{source}' is unavailable")
                                })?;
                                values.insert(output.clone(), clone_value(tensor)?);
                            }
                        }
                    }
                    ComponentImplementation::Adapter { abi, version, .. } => {
                        for value in inputs.values() {
                            self.materialize_workflow_value(values, value)?;
                        }
                        if let Some(execute) =
                            workflow_adapter_registry().get(&(abi.as_str(), version.as_str()))
                        {
                            let mut component_symbols = HashMap::new();
                            execute(
                                self,
                                component,
                                inputs,
                                outputs,
                                declaration,
                                values,
                                symbols,
                                &mut component_symbols,
                            )?;
                        } else {
                            anyhow::bail!(
                                "workflow adapter '{component}' requires unsupported ABI \
                                 {abi}@{version}"
                            );
                        }
                    }
                }
                telemetry.record_stage(
                    format!("component:{component}"),
                    stage_started.elapsed().as_nanos(),
                );
            }
            WorkflowNode::Loop {
                setup,
                body,
                continue_when,
                max_iterations,
                termination,
                iteration,
                carried,
                effects: _,
            } => {
                let plan = WorkflowLoopPlan {
                    setup,
                    body,
                    continue_when,
                    max_iterations,
                    termination,
                    iteration,
                    carried,
                };
                let limit = self.begin_workflow_loop(
                    &plan,
                    workflow,
                    values,
                    symbols,
                    dynamic_symbols,
                    emit_counts,
                    final_state_refs,
                    component_overrides,
                    telemetry,
                    host,
                )?;
                for index in 0..limit {
                    if !self.advance_workflow_loop(
                        &plan,
                        workflow,
                        index,
                        values,
                        symbols,
                        dynamic_symbols,
                        emit_counts,
                        final_state_refs,
                        component_overrides,
                        telemetry,
                        host,
                    )? {
                        break;
                    }
                }
                end_workflow_loop(&plan, values);
            }
            WorkflowNode::Branch {
                predicate,
                cases,
                default,
                outputs,
                effects: _,
            } => {
                self.materialize_workflow_value(values, predicate)?;
                let key = workflow_scalar_key(values, predicate)?;
                let (selected, is_default) = if let Some(case) = cases.get(&key) {
                    (case, false)
                } else if let Some(default) = default {
                    (default.as_ref(), true)
                } else {
                    anyhow::bail!("workflow branch has no case '{key}' and no default");
                };
                let mut device_values = Vec::new();
                for (name, value) in values.iter() {
                    if !value.is_host_resident()? {
                        device_values.push(name.clone());
                    }
                }
                for name in device_values {
                    self.materialize_workflow_value(values, &name)?;
                }
                let mut branch_values = clone_pipeline_tensors(values)?;
                let mut branch_state_refs = final_state_refs.clone();
                let emit_counts_before = emit_counts.clone();
                self.run_workflow_node(
                    selected,
                    workflow,
                    &mut branch_values,
                    symbols,
                    dynamic_symbols,
                    emit_counts,
                    &mut branch_state_refs,
                    component_overrides,
                    telemetry,
                    host,
                )?;

                // Emits are explicit side effects at the package boundary, so selected-branch
                // output values and event records survive even though ordinary case SSA does not.
                for output in workflow_emitted_outputs(selected) {
                    if let Some(value) = branch_values.get(&output) {
                        values.insert(output.clone(), clone_value(value)?);
                    }
                    if let Some(rows) = telemetry.row_outputs.get(&output) {
                        for row_output in rows {
                            if let Some(value) = branch_values.get(row_output) {
                                values.insert(row_output.clone(), clone_value(value)?);
                            }
                            let start = emit_counts_before
                                .get(row_output)
                                .copied()
                                .unwrap_or_default();
                            let end = emit_counts.get(row_output).copied().unwrap_or_default();
                            for index in start..end {
                                let event = format!("{row_output}.{index}");
                                if let Some(value) = branch_values.get(&event) {
                                    values.insert(event, clone_value(value)?);
                                }
                            }
                        }
                    }
                    let start = emit_counts_before.get(&output).copied().unwrap_or_default();
                    let end = emit_counts.get(&output).copied().unwrap_or_default();
                    for index in start..end {
                        let event = format!("{output}.{index}");
                        if let Some(value) = branch_values.get(&event) {
                            values.insert(event, clone_value(value)?);
                        }
                    }
                }

                for (output, phi) in outputs {
                    let source = if is_default {
                        phi.default.as_ref().with_context(|| {
                            format!("workflow branch output '{output}' has no default value")
                        })?
                    } else {
                        phi.cases.get(&key).with_context(|| {
                            format!(
                                "workflow branch output '{output}' has no value for case '{key}'"
                            )
                        })?
                    };
                    let value = branch_values.get(source).with_context(|| {
                        format!(
                            "workflow branch output '{output}' selected unavailable value \
                             '{source}'"
                        )
                    })?;
                    values.insert(output.clone(), clone_value(value)?);
                    for (cell, state_ref) in &branch_state_refs {
                        if state_ref == source {
                            final_state_refs.insert(cell.clone(), output.clone());
                        }
                    }
                }
            }
            WorkflowNode::Emit {
                value,
                when,
                valid_length,
                output,
                mode,
                axis,
                ..
            } => {
                let emit_started = std::time::Instant::now();
                if let Some(when) = when {
                    self.materialize_workflow_value(values, when)?;
                }
                if let Some(valid_length) = valid_length {
                    self.materialize_workflow_value(values, valid_length)?;
                }
                let tensor = {
                    let source = values
                        .get(value)
                        .with_context(|| format!("workflow emit value '{value}' is unavailable"))?;
                    if source.is_host_resident()? && self.movable_emit_values.contains(value) {
                        values.remove(value).with_context(|| {
                            format!("workflow emit value '{value}' is unavailable")
                        })?
                    } else {
                        self.materialize_workflow_value_copy(source)?
                    }
                };
                let output_contract = workflow.outputs.get(output).with_context(|| {
                    format!("workflow emit references undeclared output '{output}'")
                })?;
                let guards = when
                    .as_deref()
                    .map(|guard| workflow_bool_rows(values, guard))
                    .transpose()?;
                let lengths = valid_length
                    .as_deref()
                    .map(|length| workflow_usize_rows(values, length))
                    .transpose()?;
                // Row-wise emission is derived from structure, never from
                // serialized row identities: an output is row-wise when some
                // emit into it is ragged, and the declared request axis says
                // which axis the rows lie on. A dense request-aligned output
                // with no ragged emit is still emitted as one tensor.
                if self.row_wise_outputs.contains(output)
                    && output_contract
                        .contract
                        .batch_layout
                        .request_axis()
                        .is_some()
                {
                    emit_workflow_rows(
                        values,
                        &tensor,
                        value,
                        output,
                        &output_contract.contract,
                        mode,
                        guards.as_deref(),
                        lengths.as_deref(),
                        *axis,
                        emit_counts,
                        telemetry,
                        symbols,
                        dynamic_symbols,
                    )?;
                    telemetry.record_stage("emit", emit_started.elapsed().as_nanos());
                    return Ok(());
                }
                if guards
                    .as_ref()
                    .is_some_and(|guards| !guards.first().copied().unwrap_or(false))
                {
                    telemetry.record_stage("emit", emit_started.elapsed().as_nanos());
                    return Ok(());
                }
                let emitted = if let Some(valid_length) = valid_length {
                    let length = lengths
                        .as_ref()
                        .and_then(|lengths| lengths.first())
                        .copied()
                        .with_context(|| {
                            format!("workflow emit valid_length '{valid_length}' is invalid")
                        })?;
                    slice_workflow_prefix(&tensor, length, *axis)?
                } else {
                    tensor
                };
                if let Some(started) = telemetry.started {
                    let emitted_at = started.elapsed().as_nanos();
                    telemetry.first_emit_ns.get_or_insert(emitted_at);
                    telemetry.emit_timestamps_ns.push(emitted_at);
                }
                telemetry.emit_events += 1;
                telemetry.emitted_elements += emitted.numel() as u64;
                let validation_contract =
                    if valid_length.is_some() || matches!(mode, WorkflowEmitMode::Append) {
                        emit_chunk_contract(&output_contract.contract, &emitted, *axis)?
                    } else {
                        output_contract.contract.clone()
                    };
                validate_workflow_value(
                    value,
                    &emitted,
                    &validation_contract,
                    symbols,
                    dynamic_symbols,
                )?;
                match mode {
                    WorkflowEmitMode::Replace => {
                        values.insert(output.clone(), emitted);
                    }
                    WorkflowEmitMode::Append => {
                        let appended = if let Some(previous) = values.get(output) {
                            append_workflow_value(previous, &emitted, append_axis(*axis, &emitted))?
                        } else {
                            emitted
                        };
                        values.insert(output.clone(), appended);
                    }
                    WorkflowEmitMode::Event => {
                        let index = emit_counts.entry(output.clone()).or_default();
                        values.insert(format!("{output}.{index}"), clone_value(&emitted)?);
                        *index += 1;
                        values.insert(output.clone(), emitted);
                    }
                }
                telemetry.record_stage("emit", emit_started.elapsed().as_nanos());
            }
            WorkflowNode::Transfer {
                input,
                output,
                device,
            } => {
                if *device != DeviceKind::Cpu {
                    anyhow::bail!(
                        "workflow transfer to {device:?} requires a device allocator contract"
                    );
                }
                let tensor = values
                    .get(input)
                    .with_context(|| format!("workflow transfer value '{input}' is unavailable"))?;
                values.insert(output.clone(), clone_value(tensor)?);
            }
            WorkflowNode::ExecutionIsland { id } => {
                let stage_started = std::time::Instant::now();
                let island = self.execution_islands.get(*id).with_context(|| {
                    format!("workflow references unknown execution island {id}")
                })?;
                if island.uses_override(component_overrides) {
                    self.run_workflow_node(
                        island.fallback(),
                        workflow,
                        values,
                        symbols,
                        dynamic_symbols,
                        emit_counts,
                        final_state_refs,
                        component_overrides,
                        telemetry,
                        host,
                    )?;
                    return Ok(());
                }
                // Fusing components into one session does not make their writes
                // safe. The island's fallback still carries the original
                // bindings, so the same bounds check applies to the fused form.
                let fused = island.fallback().clone();
                for (component, inputs) in invoke_bindings(&fused) {
                    self.enforce_fixed_capacity_writes(workflow, component, inputs, values)?;
                }
                let island = self.execution_islands.get(*id).with_context(|| {
                    format!("workflow references unknown execution island {id}")
                })?;
                telemetry.component_invocations += island.component_count() as u64;
                island.run(values, component_overrides)?;
                telemetry.record_stage(format!("island:{id}"), stage_started.elapsed().as_nanos());
            }
        }
        Ok(())
    }

    /// Enforce a fixed-capacity state group's write contract before the graph runs.
    ///
    /// An `indexed_scatter` group writes to positions carried in data, so the
    /// graph itself cannot be trusted to stay inside its buffer: an out-of-range
    /// destination is a silent out-of-bounds write on most execution providers,
    /// not an error. The declaration exists precisely so the runtime can check
    /// the writes it is about to perform, and it must do so before the invoke,
    /// not after.
    ///
    /// This does not allocate, place, or size anything. Capacity is read from
    /// the graph-visible value the package declared; the buffers are whatever
    /// the caller already bound.
    fn enforce_fixed_capacity_writes(
        &self,
        workflow: &WorkflowSpec,
        component: &str,
        inputs: &BTreeMap<String, String>,
        values: &mut PipelineTensors,
    ) -> anyhow::Result<()> {
        let Some(serving) = workflow.serving.as_ref() else {
            return Ok(());
        };
        for (group_name, group) in &serving.state_service.groups {
            let Some(onnx_genai_metadata::StateUpdate::IndexedScatter {
                capacity,
                write_indices_ports,
                ..
            }) = &group.update
            else {
                continue;
            };
            let Some(port) = write_indices_ports.get(component) else {
                continue;
            };
            let Some(destinations) = inputs.get(port) else {
                continue;
            };

            self.materialize_workflow_value(values, capacity)?;
            let capacity_rows = workflow_scalar_usize(values, capacity).with_context(|| {
                format!("state service group '{group_name}' capacity '{capacity}'")
            })?;
            anyhow::ensure!(
                capacity_rows > 0,
                "state service group '{group_name}' declares capacity '{capacity}' = 0, so no \
                 write destination is legal"
            );
            let axis = group.sequence_axis.with_context(|| {
                format!(
                    "state service group '{group_name}' uses indexed_scatter without a \
                     sequence_axis"
                )
            })?;

            // The buffers must actually be the size the destinations were
            // computed against. A capacity that disagrees with the bound tensor
            // means either the declaration or the caller is wrong, and guessing
            // which would turn a detectable mismatch into a memory error.
            for (cell_name, alias) in group.ports.get(component).into_iter().flatten() {
                let Some(value_name) = inputs.get(&alias.input) else {
                    continue;
                };
                let Some(tensor) = values.get(value_name) else {
                    continue;
                };
                let shape = tensor.shape();
                let Some(extent) = shape
                    .get(axis)
                    .and_then(|extent| usize::try_from(*extent).ok())
                else {
                    anyhow::bail!(
                        "state service group '{group_name}' declares sequence_axis {axis} but \
                         state '{cell_name}' value '{value_name}' has shape {shape:?}"
                    );
                };
                anyhow::ensure!(
                    extent == capacity_rows,
                    "state service group '{group_name}' state '{cell_name}' value '{value_name}' \
                     has extent {extent} along sequence_axis {axis}, but the group declares a \
                     fixed capacity of {capacity_rows}"
                );
            }

            self.materialize_workflow_value(values, destinations)?;
            let destination_rows = values
                .get(destinations)
                .with_context(|| {
                    format!(
                        "state service group '{group_name}' write destinations '{destinations}' \
                         are unavailable"
                    )
                })?
                .to_vec_i64()
                .with_context(|| {
                    format!(
                        "state service group '{group_name}' write destinations '{destinations}' \
                         must be an integer vector"
                    )
                })?;
            for (row, destination) in destination_rows.iter().enumerate() {
                anyhow::ensure!(
                    *destination >= 0 && (*destination as u128) < capacity_rows as u128,
                    "state service group '{group_name}' would write row {row} to position \
                     {destination}, outside its capacity of {capacity_rows}"
                );
            }

            // The valid prefix is what downstream reads mask against, so a
            // length past the buffer exposes uninitialized slots as if they
            // were history.
            if let Some(lengths) = group
                .logical_lengths
                .as_deref()
                .filter(|lengths| values.contains_key(*lengths))
            {
                self.materialize_workflow_value(values, lengths)?;
                let logical = values
                    .get(lengths)
                    .with_context(|| {
                        format!(
                            "state service group '{group_name}' logical_lengths '{lengths}' \
                             are unavailable"
                        )
                    })?
                    .to_vec_i64()
                    .with_context(|| {
                        format!(
                            "state service group '{group_name}' logical_lengths '{lengths}' \
                             must be an integer vector"
                        )
                    })?;
                for (row, length) in logical.iter().enumerate() {
                    anyhow::ensure!(
                        *length >= 0 && (*length as u128) <= capacity_rows as u128,
                        "state service group '{group_name}' declares row {row} valid to \
                         length {length}, beyond its capacity of {capacity_rows}"
                    );
                }
                anyhow::ensure!(
                    logical.len() == destination_rows.len(),
                    "state service group '{group_name}' has {} logical lengths but {} write \
                     destinations; every row writes on every step",
                    logical.len(),
                    destination_rows.len()
                );
            }
        }
        Ok(())
    }

    /// Invoke one declared ONNX component with caller-supplied port values.
    ///
    /// This is the same execution seam a `WorkflowNode::Invoke` uses — port
    /// contract validation, symbol binding, resolvable output shapes, and
    /// [`Self::invoke_onnx_component`] — exposed for interpreter constructs
    /// that drive a component outside the SSA step list, such as the chained
    /// speculative proposal loop (`pipeline::speculative`). Keeping them on one
    /// seam is what makes those constructs backend-neutral: ORT and native both
    /// run through the identical path, with no second "invoke a component"
    /// implementation to keep in sync.
    ///
    /// `symbol_hints` binds shape symbols the caller has already proven for
    /// this component but that none of the inputs it is binding carry — a
    /// vocabulary an output declares and no input mentions, above all. Without
    /// them such an output has no resolvable shape, so no device buffer can be
    /// sized for it and the run has to hand it back through host memory: a
    /// per-invocation download of the very tensor the caller then wants scored
    /// on the device.
    ///
    /// A hint only ever *adds* resolution; it cannot change a shape the inputs
    /// already determine, because a symbol an input carries is re-bound from
    /// that input's actual extent and a disagreement is a validation error.
    /// And a hint that is simply wrong fails closed rather than corrupting
    /// data: an output binding is sized from the resolved shape and the session
    /// requires it to equal the kernel's own output shape exactly, so a
    /// mis-sized buffer is a run error naming both shapes, never a short read.
    pub(crate) fn invoke_component_values(
        &self,
        component: &str,
        inputs: &[(&str, &Value)],
        outputs: &std::collections::BTreeMap<String, String>,
        symbol_hints: &HashMap<String, i64>,
    ) -> anyhow::Result<Vec<(String, Value)>> {
        let workflow = &self.workflow;
        let declaration = workflow
            .components
            .get(component)
            .with_context(|| format!("workflow component '{component}' is undeclared"))?;
        anyhow::ensure!(
            matches!(
                declaration.implementation,
                ComponentImplementation::Onnx { .. }
            ),
            "workflow component '{component}' is not an ONNX component"
        );
        let mut component_symbols = symbol_hints.clone();
        let component_dynamic_symbols = std::collections::HashSet::new();
        for (port, value) in inputs {
            if let Some(contract) = declaration.ports.inputs.get(*port) {
                validate_workflow_value(
                    port,
                    value,
                    contract,
                    &mut component_symbols,
                    &component_dynamic_symbols,
                )?;
            }
        }
        let output_shapes = resolve_component_output_shapes(
            declaration,
            outputs,
            &component_symbols,
            &component_dynamic_symbols,
        );
        let produced = self.invoke_onnx_component(
            workflow,
            component,
            component,
            declaration,
            &component_symbols,
            inputs,
            outputs,
            &output_shapes,
        )?;
        for (port, tensor) in &produced {
            if let Some(contract) = declaration.ports.outputs.get(port) {
                validate_workflow_value(
                    port,
                    tensor,
                    contract,
                    &mut component_symbols,
                    &component_dynamic_symbols,
                )?;
            }
        }
        Ok(produced)
    }

    /// Backend-neutral execution seam for a declared ONNX component.
    ///
    /// The interpreter is universal over workflows; this is the one place it is
    /// universal over *backends*. It threads `onnx_genai_ort::Value` either way
    /// — the ORT branch runs the loaded `Session` (preserving the CUDA-graph /
    /// `IoBinding` stable-island fast path), and the native branch runs the
    /// pure-Rust `InferenceSession` through the `Value ⇄ Tensor` bridge. No
    /// silent fallback: a Native request that reaches here is honored natively
    /// or fails with an actionable error (Rule 4). See
    /// `docs/architecture/NATIVE_WORKFLOW_BACKEND.md`.
    #[allow(clippy::too_many_arguments)]
    fn invoke_onnx_component(
        &self,
        workflow: &WorkflowSpec,
        component: &str,
        selected_component: &str,
        selected_declaration: &onnx_genai_metadata::WorkflowComponent,
        component_symbols: &HashMap<String, i64>,
        resolved: &[(&str, &Value)],
        selected_outputs: &std::collections::BTreeMap<String, String>,
        output_shapes: &std::collections::BTreeMap<String, Vec<usize>>,
    ) -> anyhow::Result<Vec<(String, Value)>> {
        match self.decode_backend {
            EngineDecodeBackend::Native => {
                #[cfg(feature = "native-backend")]
                {
                    let _ = (workflow, component, selected_declaration, component_symbols);
                    let set = self.native_components.as_ref().with_context(|| {
                        format!(
                            "native workflow backend selected but ONNX component \
                             '{selected_component}' has no native session"
                        )
                    })?;
                    set.borrow_mut().run_component(
                        selected_component,
                        resolved,
                        selected_outputs,
                        output_shapes,
                    )
                }
                #[cfg(not(feature = "native-backend"))]
                {
                    let _ = (
                        workflow,
                        component,
                        selected_component,
                        selected_declaration,
                        component_symbols,
                        resolved,
                        selected_outputs,
                        output_shapes,
                    );
                    unreachable!(
                        "validate_pipeline_backend_request rejects Native without the \
                         native-backend feature"
                    )
                }
            }
            EngineDecodeBackend::Ort | EngineDecodeBackend::Auto => {
                let _ = output_shapes;
                let session = self.models.session(selected_component).with_context(|| {
                    format!(
                        "workflow ONNX component '{selected_component}' selected for '{component}' \
                         was not loaded"
                    )
                })?;
                let stable_eligible = session.cuda_device_id().is_some()
                    && self.device_bridge_components.contains(selected_component);
                if stable_eligible {
                    self.run_stable_component(
                        workflow,
                        selected_component,
                        selected_declaration,
                        component_symbols,
                        session,
                        resolved,
                        selected_outputs,
                    )
                } else {
                    Ok(session
                        .output_names()
                        .iter()
                        .cloned()
                        .zip(session.run(resolved)?)
                        .collect())
                }
            }
        }
    }

    // The stable-binding cache hands out `Arc<Value>` because that is the owner
    // type `Value::alias_from_shared_owner` requires. ORT values and allocators
    // are `!Send`/`!Sync` raw handles, and this cache lives behind a `RefCell`
    // on a single-threaded pipeline, so the refcount is never contended.
    #[allow(clippy::arc_with_non_send_sync)]
    #[allow(clippy::too_many_arguments)]
    fn run_stable_component(
        &self,
        workflow: &WorkflowSpec,
        component: &str,
        declaration: &onnx_genai_metadata::WorkflowComponent,
        component_symbols: &HashMap<String, i64>,
        session: &Session,
        resolved: &[(&str, &Value)],
        selected_outputs: &std::collections::BTreeMap<String, String>,
    ) -> anyhow::Result<Vec<(String, Value)>> {
        let device_id = session
            .cuda_device_id()
            .context("stable component execution requires a CUDA session")?;
        let key = (
            component.to_string(),
            resolved
                .iter()
                .map(|(name, value)| ((*name).to_string(), value.shape().to_vec()))
                .collect(),
        );
        // Shape-changing policy components (for example an append recurrence)
        // intentionally expose an output symbol that is not bound by their
        // inputs. Such outputs cannot use a fixed-address CUDA binding: ORT
        // must allocate the newly discovered shape on each invocation.
        if selected_outputs.keys().any(|output| {
            declaration
                .ports
                .outputs
                .get(output)
                .is_some_and(|contract| {
                    resolve_workflow_shape(contract, component_symbols).is_err()
                })
        }) {
            let mut values = session
                .output_names()
                .iter()
                .cloned()
                .zip(session.run(resolved)?)
                .collect::<HashMap<_, _>>();
            return selected_outputs
                .keys()
                .map(|output| {
                    values
                        .remove(output)
                        .map(|value| (output.clone(), value))
                        .with_context(|| {
                            format!(
                                "dynamic component '{component}' did not return selected output \
                                 '{output}'"
                            )
                        })
                })
                .collect();
        }
        let shared = workflow
            .serving
            .as_ref()
            .map(|serving| {
                serving
                    .state_service
                    .groups
                    .values()
                    .filter(|group| group.aliasing != StateAliasing::Forbidden)
                    .filter_map(|group| group.ports.get(component))
                    .flat_map(|aliases| aliases.values())
                    .filter(|alias| alias.access == StatePortAccess::ReadWrite)
                    .filter_map(|alias| Some((alias.output.clone()?, alias.input.clone())))
                    .collect::<HashMap<_, _>>()
            })
            .unwrap_or_default();
        let generation = self.workflow_execution_generation.get();
        let mut bindings = self.component_bindings.borrow_mut();
        if let Some(stable) = bindings.get_mut(&key) {
            let reset_services = stable.service_generation != generation;
            for (name, source) in resolved {
                let (_, destination) = stable
                    .inputs
                    .iter()
                    .find(|(input, _)| input == name)
                    .with_context(|| {
                        format!("stable component '{component}' lost input '{name}'")
                    })?;
                let shared_input = shared.values().any(|input| input == name);
                if shared_input && !reset_services {
                    continue;
                }
                if source.numel() == 0 {
                    continue;
                }
                if source.data_ptr_addr()? == destination.data_ptr_addr()? {
                    continue;
                }
                if !source.is_host_resident()? {
                    anyhow::ensure!(
                        source.device_id()? == device_id,
                        "stable component '{component}' input '{name}' is on CUDA device {}, \
                         expected {device_id}",
                        source.device_id()?
                    );
                }
                destination.copy_from_cuda(source, device_id)?;
            }
            stable.service_generation = generation;
            if session.graph_capture() {
                if !stable.captured {
                    session.synchronize_device()?;
                }
                session.run_with_binding_graph(&stable.binding, stable.graph_id)?;
                stable.captured = true;
            } else {
                session.run_with_binding(&stable.binding)?;
            }
            return stable_component_outputs(stable);
        }

        let allocator =
            if let Some(allocator) = self.component_allocators.borrow().get(component).cloned() {
                allocator
            } else {
                let allocator = Arc::new(
                    session
                        .device_kv_allocator()?
                        .context("stable component execution requires a CUDA allocator")?,
                );
                self.component_allocators
                    .borrow_mut()
                    .insert(component.to_string(), Arc::clone(&allocator));
                allocator
            };
        let discovered = if selected_outputs.keys().any(|output| {
            declaration
                .ports
                .outputs
                .get(output)
                .is_none_or(|contract| resolve_workflow_shape(contract, component_symbols).is_err())
        }) {
            let mut values = session
                .output_names()
                .iter()
                .cloned()
                .zip(session.run(resolved)?)
                .collect::<HashMap<_, _>>();
            if selected_outputs
                .keys()
                .any(|output| values.get(output).is_some_and(|value| value.numel() == 0))
            {
                return session
                    .output_names()
                    .iter()
                    .filter(|output| selected_outputs.contains_key(*output))
                    .map(|output| {
                        values
                            .remove(output)
                            .map(|value| (output.clone(), value))
                            .with_context(|| {
                                format!(
                                    "component '{component}' shape discovery did not return \
                                     selected output '{output}'"
                                )
                            })
                    })
                    .collect();
            }
            Some(values)
        } else {
            None
        };
        let mut binding = IoBinding::new(session)?;
        let mut inputs = Vec::with_capacity(resolved.len());
        for (name, source) in resolved {
            let stable = if source.numel() == 0 {
                Arc::new(Value::empty_in(source.shape(), source.dtype(), &allocator)?)
            } else if !source.is_host_resident()?
                && source.device_id()? == device_id
                && let Some(alias) = source.try_alias_clone()
            {
                Arc::new(alias?)
            } else {
                let stable = Arc::new(Value::empty_in(source.shape(), source.dtype(), &allocator)?);
                if source.numel() != 0 {
                    if !source.is_host_resident()? {
                        anyhow::ensure!(
                            source.device_id()? == device_id,
                            "stable component '{component}' input '{name}' is on CUDA device {}, \
                             expected {device_id}",
                            source.device_id()?
                        );
                    }
                    stable.copy_from_cuda(source, device_id)?;
                }
                stable
            };
            binding.bind_input(name, stable.as_ref())?;
            inputs.push(((*name).to_string(), stable));
        }
        let mut shared_outputs = Vec::new();
        let mut outputs = Vec::new();
        for output in session
            .output_names()
            .iter()
            .filter(|output| selected_outputs.contains_key(*output))
        {
            if let Some(input_name) = shared.get(output) {
                let (input_index, (_, input)) = inputs
                    .iter()
                    .enumerate()
                    .find(|(_, (name, _))| name == input_name)
                    .with_context(|| {
                        format!(
                            "stable component '{component}' shared output '{output}' has no input \
                             '{input_name}'"
                        )
                    })?;
                binding.bind_output(output, input.as_ref())?;
                shared_outputs.push((output.clone(), input_index));
            } else {
                let metadata = session
                    .outputs()
                    .iter()
                    .find(|metadata| metadata.name == *output)
                    .with_context(|| {
                        format!("stable component '{component}' output '{output}' has no metadata")
                    })?;
                let shape = if let Some(contract) = declaration.ports.outputs.get(output) {
                    resolve_workflow_shape(contract, component_symbols).or_else(|_| {
                        discovered
                            .as_ref()
                            .and_then(|values| values.get(output))
                            .map(|value| value.shape().to_vec())
                            .with_context(|| {
                                format!(
                                    "stable component '{component}' output '{output}' shape \
                                     discovery did not return that output"
                                )
                            })
                    })?
                } else {
                    discovered
                        .as_ref()
                        .and_then(|values| values.get(output))
                        .with_context(|| {
                            format!(
                                "stable component '{component}' output '{output}' shape discovery \
                                 did not return that output"
                            )
                        })?
                        .shape()
                        .to_vec()
                };
                let output_key = (
                    component.to_string(),
                    output.clone(),
                    shape.clone(),
                    format!("{:?}", metadata.dtype),
                );
                let stable = if let Some(value) =
                    self.component_outputs.borrow().get(&output_key).cloned()
                {
                    value
                } else {
                    let value = Arc::new(Value::empty_in(&shape, metadata.dtype, &allocator)?);
                    self.component_outputs
                        .borrow_mut()
                        .insert(output_key, Arc::clone(&value));
                    value
                };
                binding.bind_output(output, stable.as_ref())?;
                outputs.push((output.clone(), stable));
            }
        }
        if session.graph_capture() {
            // Warm the final fixed-address binding without capturing. The next
            // equal-shape invocation owns the non-negative graph id.
            session.run_with_binding_graph(&binding, -1)?;
        } else {
            session.run_with_binding(&binding)?;
        }
        let graph_id =
            i32::try_from(bindings.len()).context("stable component CUDA graph id exceeds i32")?;
        let stable = StableComponentBinding {
            binding,
            inputs,
            outputs,
            shared_outputs,
            output_order: session
                .output_names()
                .iter()
                .filter(|output| selected_outputs.contains_key(*output))
                .cloned()
                .collect(),
            service_generation: generation,
            graph_id,
            captured: false,
            _allocator: allocator,
        };
        let produced = stable_component_outputs(&stable)?;
        bindings.insert(key, stable);
        Ok(produced)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "matches the shared workflow adapter executor ABI"
    )]
    fn run_text_assembly_adapter(
        &self,
        component: &str,
        _inputs: &BTreeMap<String, String>,
        _outputs: &BTreeMap<String, String>,
        _declaration: &onnx_genai_metadata::WorkflowComponent,
        _values: &mut PipelineTensors,
        _symbols: &HashMap<String, i64>,
        _component_symbols: &mut HashMap<String, i64>,
    ) -> anyhow::Result<()> {
        anyhow::bail!(
            "text-assembly adapter '{component}' is an API-boundary contract and cannot be \
             invoked inside a tensor workflow"
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "matches the shared workflow adapter executor ABI"
    )]
    fn run_parameter_overlay_adapter(
        &self,
        component: &str,
        inputs: &BTreeMap<String, String>,
        outputs: &BTreeMap<String, String>,
        declaration: &onnx_genai_metadata::WorkflowComponent,
        values: &mut PipelineTensors,
        _symbols: &HashMap<String, i64>,
        _component_symbols: &mut HashMap<String, i64>,
    ) -> anyhow::Result<()> {
        let contract = declaration
            .contract
            .as_ref()
            .filter(|contract| contract.id == "onnx-genai.parameter-overlay")
            .context("parameter overlay adapter is missing its versioned contract")?;
        let input_port = workflow_contract_binding(contract, "input")?;
        let output_port = workflow_contract_binding(contract, "output")?;
        let target_component = workflow_contract_parameter(contract, "component")?;
        let target_parameter = workflow_contract_parameter(contract, "parameter")?;
        let input_name = inputs.get(input_port).with_context(|| {
            format!("parameter overlay '{component}' has no input binding '{input_port}'")
        })?;
        let output_name = outputs.get(output_port).with_context(|| {
            format!("parameter overlay '{component}' has no output binding '{output_port}'")
        })?;
        let input = values
            .get(input_name)
            .with_context(|| format!("parameter overlay input '{input_name}' is unavailable"))?;
        let shape = input.shape();
        if shape.len() != 2 {
            anyhow::bail!(
                "parameter overlay input '{input_name}' must have rank 2 [batch, features], got {shape:?}"
            );
        }
        let batch = usize::try_from(shape[0]).context("adapter batch is negative")?;
        let input_features =
            usize::try_from(shape[1]).context("adapter feature dimension is negative")?;
        let source = input
            .to_vec_f32()
            .context("portable parameter overlay requires host float32 input")?;
        let context = self.active_adapter_context.borrow();
        let context = context
            .as_ref()
            .context("parameter overlay executed without a request adapter context")?;
        // The adapter context is already in physical batch-row order, so the
        // overlay indexes it positionally. A scheduler that compacts the batch
        // moves it through the mandatory RowScopedState ABI instead.
        anyhow::ensure!(
            context.rows() == batch,
            "adapter context holds {} rows but the overlay input has batch {batch}",
            context.rows()
        );
        let cache = self.adapter_cache.borrow();
        let (result, output_features) = super::adapters::apply_parameter_overlay(
            &cache,
            context,
            target_component,
            target_parameter,
            &source,
            batch,
            input_features,
        )?;
        values.insert(
            output_name.clone(),
            Value::from_slice_f32(
                &result,
                &[i64::try_from(batch)?, i64::try_from(output_features)?],
            )?,
        );
        Ok(())
    }

    // Adapter invocations carry the full component context: its declaration,
    // both port maps, the tensor pool, and the two symbol scopes it resolves
    // shapes against. Bundling them would only rename the same seven facts.
    #[allow(clippy::too_many_arguments)]
    fn run_image_preprocess_adapter(
        &self,
        component: &str,
        inputs: &std::collections::BTreeMap<String, String>,
        outputs: &std::collections::BTreeMap<String, String>,
        declaration: &onnx_genai_metadata::WorkflowComponent,
        values: &mut PipelineTensors,
        package_symbols: &HashMap<String, i64>,
        component_symbols: &mut HashMap<String, i64>,
    ) -> anyhow::Result<()> {
        let program = self
            .preprocessing
            .as_ref()
            .and_then(|spec| spec.image.as_ref())
            .with_context(|| {
                format!(
                    "workflow image preprocessing adapter '{component}' requires \
                     preprocessing.image metadata"
                )
            })?;
        let encoded_ref = inputs.get("encoded").with_context(|| {
            format!("workflow image preprocessing adapter '{component}' requires input 'encoded'")
        })?;
        let encoded = values
            .get(encoded_ref)
            .with_context(|| format!("workflow value '{encoded_ref}' is unavailable"))?;
        let encoded_contract = declaration.ports.inputs.get("encoded").with_context(|| {
            format!(
                "workflow image preprocessing adapter '{component}' has no declared input \
                 port 'encoded'"
            )
        })?;
        let component_dynamic_symbols = std::collections::HashSet::new();
        validate_workflow_value(
            encoded_ref,
            encoded,
            encoded_contract,
            component_symbols,
            &component_dynamic_symbols,
        )?;
        if encoded.dtype() != DataType::Uint8 || encoded.shape().len() != 1 {
            anyhow::bail!(
                "workflow image preprocessing adapter '{component}' input 'encoded' must be \
                 uint8 rank 1"
            );
        }
        let pixels = program
            .outputs
            .iter()
            .find(|output| output.content == "pixels")
            .context("preprocessing.image must declare a pixels output")?;
        let pixel_contract = pixels.contract.as_ref().with_context(|| {
            format!(
                "preprocessing.image output '{}' must declare a TensorContract",
                pixels.name
            )
        })?;
        let target_shape = resolve_workflow_adapter_shape(pixel_contract, package_symbols)?;
        let processor = onnx_genai_preprocess::image::ImagePreprocessor::from_input_and_program(
            &target_shape,
            program,
        )?;
        let encoded_bytes = encoded.as_raw_bytes()?;
        let mut tensors = processor
            .preprocess_encoded([encoded_bytes])?
            .tensors
            .into_iter()
            .map(|tensor| (tensor.name.clone(), tensor))
            .collect::<HashMap<_, _>>();
        for (port, value_ref) in outputs {
            let tensor = tensors.remove(value_ref).with_context(|| {
                format!(
                    "image preprocessing adapter '{component}' did not produce declared SSA \
                     output '{value_ref}'"
                )
            })?;
            let contract = declaration.ports.outputs.get(port).with_context(|| {
                format!(
                    "workflow image preprocessing adapter '{component}' has no declared output \
                     port '{port}'"
                )
            })?;
            let value = image_tensor_to_value(tensor)?;
            validate_workflow_value(
                value_ref,
                &value,
                contract,
                component_symbols,
                &component_dynamic_symbols,
            )?;
            values.insert(value_ref.clone(), value);
        }
        if !tensors.is_empty() {
            let mut unbound = tensors.into_keys().collect::<Vec<_>>();
            unbound.sort();
            anyhow::bail!(
                "image preprocessing adapter '{component}' produced outputs without workflow \
                 bindings: {}",
                unbound.join(", ")
            );
        }
        Ok(())
    }

    // Same component context as the image adapter above.
    #[allow(clippy::too_many_arguments)]
    fn run_audio_preprocess_adapter(
        &self,
        component: &str,
        inputs: &std::collections::BTreeMap<String, String>,
        outputs: &std::collections::BTreeMap<String, String>,
        declaration: &onnx_genai_metadata::WorkflowComponent,
        values: &mut PipelineTensors,
        _package_symbols: &HashMap<String, i64>,
        component_symbols: &mut HashMap<String, i64>,
    ) -> anyhow::Result<()> {
        let program = self
            .preprocessing
            .as_ref()
            .and_then(|spec| spec.audio.as_ref())
            .with_context(|| {
                format!(
                    "workflow audio preprocessing adapter '{component}' requires \
                     preprocessing.audio metadata"
                )
            })?;
        let encoded_ref = inputs.get("encoded").with_context(|| {
            format!("workflow audio preprocessing adapter '{component}' requires input 'encoded'")
        })?;
        let encoded = values
            .get(encoded_ref)
            .with_context(|| format!("workflow value '{encoded_ref}' is unavailable"))?;
        let encoded_contract = declaration.ports.inputs.get("encoded").with_context(|| {
            format!(
                "workflow audio preprocessing adapter '{component}' has no declared input \
                 port 'encoded'"
            )
        })?;
        let component_dynamic_symbols = std::collections::HashSet::new();
        validate_workflow_value(
            encoded_ref,
            encoded,
            encoded_contract,
            component_symbols,
            &component_dynamic_symbols,
        )?;
        if encoded.dtype() != DataType::Uint8 || encoded.shape().len() != 1 {
            anyhow::bail!(
                "workflow audio preprocessing adapter '{component}' input 'encoded' must be \
                 uint8 rank 1"
            );
        }
        let executable = onnx_genai_preprocess::audio::AudioProgram::from_program(program)?;
        let encoded_bytes = encoded.as_raw_bytes()?;
        let mut tensors = executable
            .run(&[encoded_bytes])?
            .into_iter()
            .map(|tensor| (tensor.name.clone(), tensor))
            .collect::<HashMap<_, _>>();
        for (port, value_ref) in outputs {
            let tensor = tensors.remove(value_ref).with_context(|| {
                format!(
                    "audio preprocessing adapter '{component}' did not produce declared SSA \
                     output '{value_ref}'"
                )
            })?;
            let contract = declaration.ports.outputs.get(port).with_context(|| {
                format!(
                    "workflow audio preprocessing adapter '{component}' has no declared output \
                     port '{port}'"
                )
            })?;
            let value = audio_tensor_to_value(tensor)?;
            validate_workflow_value(
                value_ref,
                &value,
                contract,
                component_symbols,
                &component_dynamic_symbols,
            )?;
            values.insert(value_ref.clone(), value);
        }
        if !tensors.is_empty() {
            let mut unbound = tensors.into_keys().collect::<Vec<_>>();
            unbound.sort();
            anyhow::bail!(
                "audio preprocessing adapter '{component}' produced outputs without workflow \
                 bindings: {}",
                unbound.join(", ")
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_grammar_guidance_adapter(
        &self,
        component: &str,
        inputs: &std::collections::BTreeMap<String, String>,
        outputs: &std::collections::BTreeMap<String, String>,
        declaration: &onnx_genai_metadata::WorkflowComponent,
        values: &mut PipelineTensors,
        _package_symbols: &HashMap<String, i64>,
        component_symbols: &mut HashMap<String, i64>,
    ) -> anyhow::Result<()> {
        let contract = declaration
            .contract
            .as_ref()
            .filter(|contract| contract.id == "onnx-genai.grammar-guidance")
            .context("grammar guidance adapter is missing its versioned contract")?;
        let action = workflow_contract_parameter(contract, "action")?;
        if !matches!(action, "clone" | "lookahead" | "commit") {
            anyhow::bail!("workflow grammar adapter has unknown action '{action}'");
        }
        let state = workflow_contract_binding(contract, "state")?;
        let tokens = workflow_contract_binding(contract, "tokens")?;
        let valid_length = workflow_contract_binding(contract, "valid_length")?;
        let transition_table = workflow_contract_binding(contract, "transition_table")?;
        let next_state = workflow_contract_binding(contract, "next_state")?;
        let consumed_length = workflow_contract_binding(contract, "consumed_length")?;
        let logits_mask = workflow_contract_binding(contract, "logits_mask")?;
        let forced_tokens = workflow_contract_binding(contract, "forced_tokens")?;
        let forced_length = workflow_contract_binding(contract, "forced_length")?;

        let state_value = workflow_adapter_input(component, state, inputs, values)?;
        let token_value = workflow_adapter_input(component, tokens, inputs, values)?;
        let length_value = workflow_adapter_input(component, valid_length, inputs, values)?;
        let table_value = workflow_adapter_input(component, transition_table, inputs, values)?;
        let states = state_value.to_vec_i64()?;
        let token_data = token_value.to_vec_i64()?;
        let lengths = length_value.to_vec_i64()?;
        let transitions = table_value.to_vec_i64()?;
        let batch = states.len();
        let token_shape = token_value.shape();
        let table_shape = table_value.shape();
        if state_value.shape() != [batch as i64]
            || token_shape.len() != 2
            || token_shape[0] != batch as i64
            || length_value.shape() != [batch as i64]
            || table_shape.len() != 2
        {
            anyhow::bail!(
                "workflow grammar adapter '{component}' received incompatible runtime shapes"
            );
        }
        let token_width = usize::try_from(token_shape[1])?;
        let state_count = usize::try_from(table_shape[0])?;
        let vocabulary = usize::try_from(table_shape[1])?;
        let transition_count = state_count
            .checked_mul(vocabulary)
            .context("workflow grammar transition-table shape overflows usize")?;
        if transitions.len() != transition_count {
            anyhow::bail!(
                "workflow grammar adapter '{component}' transition table size is invalid"
            );
        }

        let mut next_states = Vec::with_capacity(batch);
        let mut consumed = Vec::with_capacity(batch);
        let mut mask = Vec::with_capacity(batch * vocabulary);
        let mut forced = Vec::with_capacity(batch);
        let mut forced_lengths = Vec::with_capacity(batch);
        for row in 0..batch {
            let requested = if action == "clone" {
                0
            } else {
                usize::try_from(lengths[row]).with_context(|| {
                    format!("grammar valid_length for row {row} must be non-negative")
                })?
            };
            if requested > token_width {
                anyhow::bail!(
                    "workflow grammar adapter '{component}' row {row} valid_length {requested} \
                     exceeds token width {token_width}"
                );
            }
            let mut state_index = usize::try_from(states[row])
                .with_context(|| format!("grammar state for row {row} must be non-negative"))?;
            if state_index >= state_count {
                anyhow::bail!(
                    "workflow grammar adapter '{component}' row {row} state {state_index} is \
                     outside {state_count} states"
                );
            }
            let mut accepted = 0usize;
            for column in 0..requested {
                let token = usize::try_from(token_data[row * token_width + column])
                    .with_context(|| format!("grammar token for row {row} must be non-negative"))?;
                let next = token
                    .checked_add(state_index * vocabulary)
                    .filter(|index| *index < transitions.len())
                    .map(|index| transitions[index])
                    .unwrap_or(-1);
                if next < 0 {
                    if action == "commit" {
                        anyhow::bail!(
                            "workflow grammar commit adapter '{component}' rejected token {token} \
                             at row {row}, position {column}"
                        );
                    }
                    break;
                }
                state_index = usize::try_from(next)?;
                if state_index >= state_count {
                    anyhow::bail!(
                        "workflow grammar adapter '{component}' transition produced invalid \
                         state {state_index}"
                    );
                }
                accepted += 1;
            }
            next_states.push(i64::try_from(state_index)?);
            consumed.push(i64::try_from(accepted)?);
            let row_transitions =
                &transitions[state_index * vocabulary..(state_index + 1) * vocabulary];
            let mut only_token = None;
            for (token, next) in row_transitions.iter().enumerate() {
                let allowed = *next >= 0;
                mask.push(u8::from(allowed));
                if allowed {
                    only_token = match only_token {
                        None => Some(token),
                        Some(_) => Some(vocabulary),
                    };
                }
            }
            if let Some(token) = only_token.filter(|token| *token < vocabulary) {
                forced.push(i64::try_from(token)?);
                forced_lengths.push(1);
            } else {
                forced.push(0);
                forced_lengths.push(0);
            }
        }

        let produced = [
            (
                next_state,
                Value::from_slice_i64(&next_states, &[batch as i64])?,
            ),
            (
                consumed_length,
                Value::from_slice_i64(&consumed, &[batch as i64])?,
            ),
            (
                logits_mask,
                Value::from_raw_bytes(mask, &[batch as i64, vocabulary as i64], DataType::Bool)?,
            ),
            (
                forced_tokens,
                Value::from_slice_i64(&forced, &[batch as i64, 1])?,
            ),
            (
                forced_length,
                Value::from_slice_i64(&forced_lengths, &[batch as i64])?,
            ),
        ];
        for (port, value) in produced {
            insert_workflow_adapter_output(
                component,
                port,
                value,
                outputs,
                declaration,
                values,
                component_symbols,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_telemetry_adapter(
        &self,
        component: &str,
        inputs: &std::collections::BTreeMap<String, String>,
        outputs: &std::collections::BTreeMap<String, String>,
        declaration: &onnx_genai_metadata::WorkflowComponent,
        values: &mut PipelineTensors,
        _package_symbols: &HashMap<String, i64>,
        component_symbols: &mut HashMap<String, i64>,
    ) -> anyhow::Result<()> {
        static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        let elapsed_ns = || {
            i64::try_from(
                EPOCH
                    .get_or_init(std::time::Instant::now)
                    .elapsed()
                    .as_nanos(),
            )
            .context("workflow telemetry timestamp exceeds int64")
        };
        let contract = declaration
            .contract
            .as_ref()
            .filter(|contract| contract.id == "onnx-genai.telemetry")
            .context("telemetry adapter is missing its versioned contract")?;
        match workflow_contract_parameter(contract, "action")? {
            "start" => {
                let port = workflow_contract_binding(contract, "timestamp")?;
                insert_workflow_adapter_output(
                    component,
                    port,
                    Value::from_slice_i64(&[elapsed_ns()?], &[])?,
                    outputs,
                    declaration,
                    values,
                    component_symbols,
                )
            }
            "elapsed" => {
                let timestamp_port = workflow_contract_binding(contract, "timestamp")?;
                let started = workflow_adapter_input(component, timestamp_port, inputs, values)?
                    .to_vec_i64()?;
                let [started] = started.as_slice() else {
                    anyhow::bail!("workflow telemetry timestamp must contain one value");
                };
                let duration = (elapsed_ns()? - *started).max(0) as f32 / 1_000_000.0;
                let port = workflow_contract_binding(contract, "duration_ms")?;
                insert_workflow_adapter_output(
                    component,
                    port,
                    Value::from_slice_f32(&[duration], &[])?,
                    outputs,
                    declaration,
                    values,
                    component_symbols,
                )
            }
            action => anyhow::bail!("workflow telemetry adapter has unknown action '{action}'"),
        }
    }

    fn bind_workflow_inputs(
        &self,
        workflow: &WorkflowSpec,
        request: &GenerateRequest,
        mut provided: PipelineTensors,
        host_supplied: &std::collections::HashSet<String>,
        runtime_managed_seeds: &std::collections::HashSet<String>,
        serving_controls: &std::collections::BTreeMap<String, super::generation::ServingControl>,
    ) -> anyhow::Result<(
        PipelineTensors,
        std::collections::BTreeMap<String, onnx_genai_metadata::LiteralValue>,
    )> {
        let mut values = HashMap::new();
        // The literal each materialized value was built from, not merely that
        // one was: a literal whose contract has symbolic axes is first
        // materialized at `literal_shape`'s singleton guess and must be rebuilt
        // once the request's own values have bound those symbols. Keeping the
        // literal here is what lets a *derived* seed be rebuilt too — it has no
        // declared default to go back to.
        let mut from_literal = std::collections::BTreeMap::new();
        for (name, input) in &workflow.inputs {
            let supplied = provided.remove(name).or_else(|| match &input.source {
                WorkflowInputSource::Application { name } => provided.remove(name),
                _ => None,
            });
            // The service that owns a runtime-managed cell owns its buffer, and
            // the declared seed for such a cell is not a tensor the interpreter
            // can build: a growing cache declares its rank but not its extent.
            // A caller that supplied one is honoured; nothing is invented for
            // one that did not.
            if supplied.is_none() && runtime_managed_seeds.contains(name) {
                continue;
            }
            let value = if let Some(value) = supplied {
                Some(value)
            } else {
                match &input.source {
                    WorkflowInputSource::Request => match &input.role {
                        onnx_genai_metadata::SemanticInputRole::Runtime { role, .. } => {
                            workflow_request_value(role, request, &input.contract)?
                        }
                        onnx_genai_metadata::SemanticInputRole::Opaque => {
                            anyhow::bail!(
                                "workflow request input '{name}' must declare a versioned runtime role"
                            )
                        }
                    },
                    WorkflowInputSource::Literal => {
                        if let Some(literal) = input.default.as_ref() {
                            from_literal.insert(name.clone(), literal.clone());
                        }
                        input
                            .default
                            .as_ref()
                            .map(|value| workflow_literal_value(value, &input.contract))
                            .transpose()?
                    }
                    WorkflowInputSource::Application { .. } => {
                        if let Some(literal) = input.default.as_ref() {
                            from_literal.insert(name.clone(), literal.clone());
                        }
                        input
                            .default
                            .as_ref()
                            .map(|value| workflow_literal_value(value, &input.contract))
                            .transpose()?
                    }
                    WorkflowInputSource::Artifact { path } => {
                        anyhow::bail!(
                            "workflow input '{name}' requires artifact binding '{path}', which \
                             is not a tensor request input"
                        )
                    }
                }
            };
            // The serving service owns its own control signals. If neither the
            // caller nor the package supplied one, seed it from the role the
            // serving contract declared rather than refusing a package whose
            // only "missing" inputs are the runtime's own.
            let value = match (value, serving_controls.get(name)) {
                (None, Some(role)) => {
                    let seed = role.admission_seed();
                    let value =
                        workflow_literal_value(&seed, &input.contract).with_context(|| {
                            format!(
                                "failed to seed serving control '{name}' from its declared \
                                 contract"
                            )
                        })?;
                    from_literal.insert(name.clone(), seed);
                    Some(value)
                }
                (value, _) => value,
            };
            let is_present = value.is_some();
            if let Some(present_as) = &input.present_as {
                values.insert(
                    present_as.clone(),
                    Value::from_raw_bytes(vec![u8::from(is_present)], &[], DataType::Bool)?,
                );
            }
            match value {
                Some(value) => {
                    values.insert(name.clone(), value);
                }
                None if input.required && !host_supplied.contains(name) => {
                    anyhow::bail!("{MISSING_REQUIRED_INPUT}'{name}' was not supplied")
                }
                None => {}
            }
        }
        if !provided.is_empty() {
            anyhow::bail!(
                "workflow request supplied undeclared application inputs: {:?}",
                provided.keys().collect::<Vec<_>>()
            );
        }
        Ok((values, from_literal))
    }
}

/// Sampling parameters as the workflow sampler must see them.
///
/// The workflow sampler is a graph: it has no access to `GenerateOptions::greedy`
/// and can only be steered through its declared `sampling_*` runtime roles. Binding
/// the raw request options would therefore run a *stochastic* sampler for a greedy
/// request, because the runtime's own defaults (`top_k = 0`, `temperature = 1.0`)
/// disable truncation entirely. The result is seeded — hence reproducible — but it
/// silently departs from argmax whenever the top-2 logits are close, which is
/// exactly where a greedy contract must not drift.
///
/// Greedy is projected onto the sampler contract as `temperature = 0` *and*
/// `top_k = 1`, with the nucleus filters opened back up so a caller-supplied
/// `top_p`/`min_p` cannot truncate the single surviving candidate. Either knob
/// alone already forces argmax for a conforming sampler; sending both keeps the
/// contract unambiguous for samplers that implement only one of them.
struct EffectiveSampling {
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
}

fn effective_sampling(request: &GenerateRequest) -> EffectiveSampling {
    let options = &request.options;
    if options.greedy {
        EffectiveSampling {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            min_p: 0.0,
        }
    } else {
        EffectiveSampling {
            temperature: options.temperature,
            top_k: options.top_k,
            top_p: options.top_p,
            min_p: options.min_p,
        }
    }
}

fn workflow_request_value(
    field: &RuntimeInputRole,
    request: &GenerateRequest,
    contract: &TensorContract,
) -> anyhow::Result<Option<Value>> {
    let scalar_i64 = |value: i64| {
        let shape = scalar_or_batch_shape(contract)?;
        Value::from_slice_i64(&vec![value; shape_numel(&shape)], &shape).map_err(Into::into)
    };
    let scalar_f32 = |value: f32| {
        let shape = scalar_or_batch_shape(contract)?;
        Value::from_slice_f32(&vec![value; shape_numel(&shape)], &shape).map_err(Into::into)
    };
    match field {
        RuntimeInputRole::PromptTokens => match &request.prompt {
            GeneratePrompt::TokenIds(tokens) => {
                let data = tokens
                    .iter()
                    .map(|token| i64::from(*token))
                    .collect::<Vec<_>>();
                let shape = match contract.rank {
                    1 => vec![data.len() as i64],
                    2 => vec![1, data.len() as i64],
                    rank => anyhow::bail!(
                        "prompt token workflow input must have rank 1 or 2, got {rank}"
                    ),
                };
                Ok(Some(Value::from_slice_i64(&data, &shape)?))
            }
            GeneratePrompt::TokenRows(rows) => {
                anyhow::ensure!(
                    contract.rank == 2,
                    "multi-row prompt token workflow input must have rank 2"
                );
                anyhow::ensure!(!rows.is_empty(), "multi-row prompt must not be empty");
                let columns = rows[0].len();
                anyhow::ensure!(
                    rows.iter().all(|row| row.len() == columns),
                    "multi-row prompt token rows must have equal lengths"
                );
                let data = rows
                    .iter()
                    .flatten()
                    .map(|token| i64::from(*token))
                    .collect::<Vec<_>>();
                Ok(Some(Value::from_slice_i64(
                    &data,
                    &[rows.len() as i64, columns as i64],
                )?))
            }
            GeneratePrompt::Text(_) => anyhow::bail!(
                "prompt_tokens request binding requires token ids; use a tokenizer adapter for text"
            ),
        },
        RuntimeInputRole::PromptText => {
            anyhow::bail!("prompt_text request binding requires a versioned tokenizer adapter")
        }
        RuntimeInputRole::NegativePromptText
        | RuntimeInputRole::NegativePromptTokens
        | RuntimeInputRole::GuidanceScale
        | RuntimeInputRole::Width
        | RuntimeInputRole::Height
        | RuntimeInputRole::DenoisingStrength => Ok(None),
        RuntimeInputRole::MaxIterations | RuntimeInputRole::MaxOutputTokens => {
            scalar_i64(request.options.max_new_tokens as i64).map(Some)
        }
        // A request may narrow the package's EOS set; when it says nothing the
        // package's own declared literal stands, which is why this yields
        // `None` rather than an empty tensor that would mean "nothing stops".
        RuntimeInputRole::EosTokenIds => match request.options.eos_token_id {
            Some(id) => Ok(Some(Value::from_slice_i64(&[i64::from(id)], &[1])?)),
            None => Ok(None),
        },
        RuntimeInputRole::Seed => {
            scalar_i64(request.options.seed.unwrap_or_default() as i64).map(Some)
        }
        RuntimeInputRole::SamplingTemperature => {
            scalar_f32(effective_sampling(request).temperature).map(Some)
        }
        RuntimeInputRole::SamplingTopK => {
            scalar_i64(effective_sampling(request).top_k as i64).map(Some)
        }
        RuntimeInputRole::SamplingTopP => scalar_f32(effective_sampling(request).top_p).map(Some),
        RuntimeInputRole::SamplingMinP => scalar_f32(effective_sampling(request).min_p).map(Some),
        RuntimeInputRole::Media
        | RuntimeInputRole::Constraint
        | RuntimeInputRole::SessionId
        | RuntimeInputRole::RowSelection
        | RuntimeInputRole::AdapterSegments
        | RuntimeInputRole::AdapterCounts
        | RuntimeInputRole::AdapterScales
        | RuntimeInputRole::AdapterActive => Ok(None),
    }
}

fn literal_element_bytes(scalar: &ScalarValue, dtype: &str) -> anyhow::Result<(Vec<u8>, DataType)> {
    match scalar {
        ScalarValue::Integer(value) => match dtype {
            "int64" => Ok((value.to_le_bytes().to_vec(), DataType::Int64)),
            "int32" => Ok((
                i32::try_from(*value)
                    .context("integer literal exceeds int32")?
                    .to_le_bytes()
                    .to_vec(),
                DataType::Int32,
            )),
            "int16" => Ok((
                i16::try_from(*value)
                    .context("integer literal exceeds int16")?
                    .to_le_bytes()
                    .to_vec(),
                DataType::Int16,
            )),
            "int8" => Ok((
                vec![i8::try_from(*value).context("integer literal exceeds int8")? as u8],
                DataType::Int8,
            )),
            "uint64" => Ok((
                u64::try_from(*value)
                    .context("integer literal is negative")?
                    .to_le_bytes()
                    .to_vec(),
                DataType::Uint64,
            )),
            "uint32" => Ok((
                u32::try_from(*value)
                    .context("integer literal exceeds uint32")?
                    .to_le_bytes()
                    .to_vec(),
                DataType::Uint32,
            )),
            "uint16" => Ok((
                u16::try_from(*value)
                    .context("integer literal exceeds uint16")?
                    .to_le_bytes()
                    .to_vec(),
                DataType::Uint16,
            )),
            "uint8" => Ok((
                vec![u8::try_from(*value).context("integer literal exceeds uint8")?],
                DataType::Uint8,
            )),
            _ => anyhow::bail!(
                "integer workflow literal is incompatible with declared dtype '{dtype}'"
            ),
        },
        ScalarValue::Float(value) => match dtype {
            "float32" | "fp32" => Ok(((*value as f32).to_le_bytes().to_vec(), DataType::Float32)),
            "float16" | "fp16" => Ok((
                half::f16::from_f64(*value).to_bits().to_le_bytes().to_vec(),
                DataType::Float16,
            )),
            "bfloat16" | "bf16" => Ok((
                half::bf16::from_f64(*value)
                    .to_bits()
                    .to_le_bytes()
                    .to_vec(),
                DataType::BFloat16,
            )),
            _ => anyhow::bail!(
                "floating-point workflow literal is incompatible with declared dtype '{dtype}'"
            ),
        },
        ScalarValue::Bool(value) if dtype == "bool" => Ok((vec![u8::from(*value)], DataType::Bool)),
        ScalarValue::String(_) => {
            anyhow::bail!("string literal workflow inputs require an adapter binding")
        }
        _ => anyhow::bail!("workflow literal is incompatible with declared dtype '{dtype}'"),
    }
}

fn workflow_literal_value(
    literal: &LiteralValue,
    contract: &TensorContract,
) -> anyhow::Result<Value> {
    let shape = literal_shape(literal, contract)?;
    workflow_literal_value_with_shape(literal, contract, &shape)
}

/// Materialize a literal across an explicitly resolved shape.
///
/// Split out from [`workflow_literal_value`] so a literal whose contract has
/// symbolic axes can be re-materialized once request inputs have bound those
/// symbols, instead of being frozen at the singleton `literal_shape` guess.
fn workflow_literal_value_with_shape(
    literal: &LiteralValue,
    contract: &TensorContract,
    shape: &[i64],
) -> anyhow::Result<Value> {
    let shape = shape.to_vec();
    let numel = shape_numel(&shape);
    match literal {
        // One scalar broadcasts to the whole tensor.
        LiteralValue::Scalar(scalar) => {
            let (bytes, dtype) = literal_element_bytes(scalar, contract.dtype.as_str())?;
            Value::from_raw_bytes(bytes.repeat(numel), &shape, dtype).map_err(Into::into)
        }
        // Explicit elements are laid out row-major and must fill the contract.
        LiteralValue::Elements(elements) => {
            if elements.len() != numel {
                anyhow::bail!(
                    "workflow literal declares {} elements but its contract holds {numel}",
                    elements.len()
                );
            }
            let mut bytes = Vec::new();
            let mut element_dtype = None;
            for element in elements {
                let (encoded, dtype) = literal_element_bytes(element, contract.dtype.as_str())?;
                if *element_dtype.get_or_insert(dtype) != dtype {
                    anyhow::bail!("workflow literal mixes element types");
                }
                bytes.extend_from_slice(&encoded);
            }
            let dtype = element_dtype.context("workflow literal element list must not be empty")?;
            Value::from_raw_bytes(bytes, &shape, dtype).map_err(Into::into)
        }
    }
}

fn scalar_or_batch_shape(contract: &TensorContract) -> anyhow::Result<Vec<i64>> {
    match contract.rank {
        0 => Ok(Vec::new()),
        1 => Ok(vec![1]),
        rank => anyhow::bail!("request scalar binding requires rank 0 or 1, got {rank}"),
    }
}

fn workflow_contract_binding<'a>(
    contract: &'a onnx_genai_metadata::ComponentContract,
    role: &str,
) -> anyhow::Result<&'a str> {
    contract
        .bindings
        .get(role)
        .map(String::as_str)
        .with_context(|| {
            format!(
                "workflow contract '{}' has no '{role}' binding",
                contract.id
            )
        })
}

fn workflow_contract_parameter<'a>(
    contract: &'a onnx_genai_metadata::ComponentContract,
    name: &str,
) -> anyhow::Result<&'a str> {
    match contract.parameters.get(name) {
        Some(ScalarValue::String(value)) => Ok(value),
        _ => anyhow::bail!(
            "workflow contract '{}' requires string parameter '{name}'",
            contract.id
        ),
    }
}

fn workflow_adapter_input<'a>(
    component: &str,
    port: &str,
    inputs: &std::collections::BTreeMap<String, String>,
    values: &'a PipelineTensors,
) -> anyhow::Result<&'a Value> {
    let value_ref = inputs
        .get(port)
        .with_context(|| format!("workflow adapter '{component}' requires input port '{port}'"))?;
    values
        .get(value_ref)
        .with_context(|| format!("workflow adapter input value '{value_ref}' is unavailable"))
}

#[allow(clippy::too_many_arguments)]
fn insert_workflow_adapter_output(
    component: &str,
    port: &str,
    value: Value,
    outputs: &std::collections::BTreeMap<String, String>,
    declaration: &onnx_genai_metadata::WorkflowComponent,
    values: &mut PipelineTensors,
    component_symbols: &mut HashMap<String, i64>,
) -> anyhow::Result<()> {
    let value_ref = outputs.get(port).with_context(|| {
        format!("workflow adapter '{component}' requires output binding for port '{port}'")
    })?;
    let contract = declaration.ports.outputs.get(port).with_context(|| {
        format!("workflow adapter '{component}' has no declared output port '{port}'")
    })?;
    validate_workflow_value(
        value_ref,
        &value,
        contract,
        component_symbols,
        &std::collections::HashSet::new(),
    )?;
    values.insert(value_ref.clone(), value);
    Ok(())
}

fn resolve_workflow_shape(
    contract: &TensorContract,
    symbols: &HashMap<String, i64>,
) -> anyhow::Result<Vec<i64>> {
    let shape = contract
        .shape
        .as_ref()
        .context("workflow adapter output requires a declared shape")?;
    shape
        .iter()
        .map(|dimension| match dimension {
            TensorDimension::Fixed(value) => Ok(*value),
            TensorDimension::Symbol(symbol) => symbols.get(symbol).copied().with_context(|| {
                format!(
                    "workflow adapter output requires unresolved dimension '{symbol}' for allocation"
                )
            }),
        })
        .collect()
}

fn resolve_workflow_adapter_shape(
    contract: &TensorContract,
    _symbols: &HashMap<String, i64>,
) -> anyhow::Result<Vec<i64>> {
    let shape = contract
        .shape
        .as_ref()
        .context("workflow adapter output requires a declared shape")?;
    shape
        .iter()
        .map(|dimension| match dimension {
            TensorDimension::Fixed(value) => Ok(*value),
            // Adapter outputs are produced before their concrete dimensions
            // exist. A symbol here is a typed dynamic axis, not permission to
            // borrow an equal-spelled extent from another component artifact
            // (for example a source image and generated latent that both call
            // their private axes "height").
            TensorDimension::Symbol(_) => Ok(-1),
        })
        .collect()
}

fn workflow_iteration_value(
    index: usize,
    contract: &TensorContract,
    symbols: &HashMap<String, i64>,
) -> anyhow::Result<Value> {
    if contract.dtype != "int64" {
        anyhow::bail!(
            "workflow loop iteration requires dtype int64, got '{}'",
            contract.dtype
        );
    }
    let index = i64::try_from(index).context("workflow loop iteration exceeds int64")?;
    match contract.rank {
        0 => Value::from_slice_i64(&[index], &[]).map_err(Into::into),
        1 => {
            let shape = resolve_workflow_shape(contract, symbols)?;
            let elements = shape_numel(&shape);
            Value::from_vec_i64(vec![index; elements], &shape).map_err(Into::into)
        }
        rank => anyhow::bail!("workflow loop iteration requires rank 0 or rank 1, got rank {rank}"),
    }
}

fn audio_tensor_to_value(
    tensor: onnx_genai_preprocess::audio::NamedAudioTensor,
) -> anyhow::Result<Value> {
    use onnx_genai_preprocess::audio::AudioTensorData;

    let shape = &tensor.shape;
    match tensor.data {
        AudioTensorData::Fp32(data) => Value::from_vec_f32(data, shape).map_err(Into::into),
        AudioTensorData::Int64(data) => Value::from_vec_i64(data, shape).map_err(Into::into),
    }
}

fn image_tensor_to_value(
    tensor: onnx_genai_preprocess::image::NamedImageTensor,
) -> anyhow::Result<Value> {
    use onnx_genai_preprocess::image::ImageTensorData;

    let shape = &tensor.shape;
    match tensor.data {
        ImageTensorData::Fp32(data) => Value::from_vec_f32(data, shape).map_err(Into::into),
        ImageTensorData::Fp16(data) => Value::from_vec_f16_bits(data, shape).map_err(Into::into),
        ImageTensorData::Bf16(data) => Value::from_vec_bf16_bits(data, shape).map_err(Into::into),
        ImageTensorData::Int64(data) => Value::from_vec_i64(data, shape).map_err(Into::into),
        ImageTensorData::Int32(data) => {
            typed_image_bytes(data, shape, DataType::Int32, i32::to_ne_bytes)
        }
        ImageTensorData::Int8(data) => Value::from_raw_bytes(
            data.into_iter().map(|value| value as u8).collect(),
            shape,
            DataType::Int8,
        )
        .map_err(Into::into),
        ImageTensorData::Uint8(data) => {
            Value::from_raw_bytes(data, shape, DataType::Uint8).map_err(Into::into)
        }
        ImageTensorData::Bool(data) => {
            Value::from_raw_bytes(data, shape, DataType::Bool).map_err(Into::into)
        }
    }
}

fn typed_image_bytes<T, const N: usize>(
    data: Vec<T>,
    shape: &[i64],
    dtype: DataType,
    to_bytes: impl Fn(T) -> [u8; N],
) -> anyhow::Result<Value> {
    let bytes = data.into_iter().flat_map(to_bytes).collect::<Vec<u8>>();
    Value::from_raw_bytes(bytes, shape, dtype).map_err(Into::into)
}

/// Concrete shape for a literal, given what the literal itself says.
///
/// A symbolic axis has no extent until something binds it. For a *scalar*
/// literal nothing does, so the axis starts as a singleton and later
/// request/component values unify the shared symbol.
///
/// An **element list is different: it carries its own extent.** A package that
/// declares `[eos_token_ids]` and lists two ids has stated the axis is 2, and
/// guessing 1 discards the only information available — which is why a
/// multi-EOS package failed with "declares 2 elements but its contract holds 1"
/// against a shape it had described correctly. When exactly one axis is
/// symbolic, the list divides out the fixed axes to size it. With more than one
/// unbound axis the split is genuinely ambiguous, so the singleton guess stands
/// and the caller gets the element-count error, which names the real problem.
fn literal_shape(literal: &LiteralValue, contract: &TensorContract) -> anyhow::Result<Vec<i64>> {
    let Some(shape) = &contract.shape else {
        if contract.rank == 0 {
            return Ok(Vec::new());
        }
        anyhow::bail!("literal workflow input requires a fully declared shape");
    };
    let symbolic_axes = shape
        .iter()
        .filter(|dimension| matches!(dimension, TensorDimension::Symbol(_)))
        .count();
    let fixed_extent: i64 = shape
        .iter()
        .filter_map(|dimension| match dimension {
            TensorDimension::Fixed(value) => Some(*value),
            TensorDimension::Symbol(_) => None,
        })
        .product();
    let symbolic_extent = match literal {
        LiteralValue::Elements(elements) if symbolic_axes == 1 && fixed_extent > 0 => {
            let elements = elements.len() as i64;
            anyhow::ensure!(
                elements % fixed_extent == 0,
                "workflow literal lists {elements} elements, which does not divide evenly \
                 across the contract's fixed axes ({fixed_extent} elements per symbolic step)"
            );
            elements / fixed_extent
        }
        _ => 1,
    };
    Ok(shape
        .iter()
        .map(|dimension| match dimension {
            TensorDimension::Fixed(value) => *value,
            TensorDimension::Symbol(_) => symbolic_extent,
        })
        .collect())
}

fn shape_numel(shape: &[i64]) -> usize {
    shape.iter().map(|dimension| *dimension as usize).product()
}

/// Resolve each declared output port's contract shape to concrete dimensions
/// using the invocation's already-bound component symbols.
///
/// Returns only ports whose every dimension is known now (a fixed dim, or a
/// symbol the inputs already bound) — a genuinely dynamic output dimension is
/// omitted, so the native CUDA seam can pre-allocate a device output buffer for
/// the resolvable ones (keeping a recurring/state tensor device-resident) and
/// fall back to host materialization for the rest. Backend-agnostic: the ORT
/// path ignores it.
fn resolve_component_output_shapes(
    declaration: &onnx_genai_metadata::WorkflowComponent,
    selected_outputs: &std::collections::BTreeMap<String, String>,
    symbols: &HashMap<String, i64>,
    dynamic_symbols: &std::collections::HashSet<String>,
) -> std::collections::BTreeMap<String, Vec<usize>> {
    let mut resolved = std::collections::BTreeMap::new();
    for port in selected_outputs.keys() {
        let Some(contract) = declaration.ports.outputs.get(port) else {
            continue;
        };
        let Some(shape) = &contract.shape else {
            continue;
        };
        let mut dims = Vec::with_capacity(shape.len());
        let mut fully_known = true;
        for dim in shape {
            match dim {
                TensorDimension::Fixed(value) if *value >= 0 => dims.push(*value as usize),
                TensorDimension::Symbol(symbol) if !dynamic_symbols.contains(symbol) => {
                    match symbols.get(symbol) {
                        Some(value) if *value >= 0 => dims.push(*value as usize),
                        _ => {
                            fully_known = false;
                            break;
                        }
                    }
                }
                _ => {
                    fully_known = false;
                    break;
                }
            }
        }
        if fully_known {
            resolved.insert(port.clone(), dims);
        }
    }
    resolved
}

fn validate_workflow_value(
    name: &str,
    value: &Value,
    contract: &TensorContract,
    symbols: &mut HashMap<String, i64>,
    dynamic_symbols: &std::collections::HashSet<String>,
) -> anyhow::Result<()> {
    let expected_dtype = match contract.dtype.as_str() {
        "float32" | "fp32" => DataType::Float32,
        "float16" | "fp16" => DataType::Float16,
        "bfloat16" | "bf16" => DataType::BFloat16,
        // FP8 state and activations are ordinary 1-byte ORT tensors. Whether a
        // kernel can COMPUTE on them is an execution-provider question answered
        // at binding time; refusing to carry the value here would turn a
        // capability gap into a false schema error.
        "float8_e4m3fn" | "fp8_e4m3fn" | "float8_e4m3" | "fp8_e4m3" => DataType::Float8E4M3,
        "float8_e5m2" | "fp8_e5m2" => DataType::Float8E5M2,
        "int64" => DataType::Int64,
        "int32" => DataType::Int32,
        "int16" => DataType::Int16,
        "int8" => DataType::Int8,
        "uint64" => DataType::Uint64,
        "uint32" => DataType::Uint32,
        "uint16" => DataType::Uint16,
        "uint8" => DataType::Uint8,
        "bool" => DataType::Bool,
        dtype => anyhow::bail!("workflow value '{name}' uses unsupported dtype '{dtype}'"),
    };
    if value.dtype() != expected_dtype {
        anyhow::bail!(
            "workflow value '{name}' has dtype {:?}, expected {}",
            value.dtype(),
            contract.dtype
        );
    }
    if value.shape().len() != contract.rank {
        anyhow::bail!(
            "workflow value '{name}' has rank {}, expected {}",
            value.shape().len(),
            contract.rank
        );
    }
    if let Some(shape) = &contract.shape {
        for (axis, (declared, actual)) in shape.iter().zip(value.shape()).enumerate() {
            let symbolic_actual = if contract.batch_layout.request_axis() == Some(axis) {
                let factor = i64::try_from(contract.batch_layout.request_expansion_factor())?;
                anyhow::ensure!(
                    factor > 0,
                    "workflow value '{name}' has a zero request expansion factor"
                );
                if actual % factor != 0 {
                    anyhow::bail!(
                        "workflow value '{name}' axis {axis} is {actual}, which is not divisible \
                         by its request expansion factor {factor}"
                    );
                }
                actual / factor
            } else {
                *actual
            };
            match declared {
                TensorDimension::Fixed(expected) if expected != actual => anyhow::bail!(
                    "workflow value '{name}' axis {axis} is {actual}, expected {expected}"
                ),
                TensorDimension::Symbol(symbol) if dynamic_symbols.contains(symbol) => {}
                TensorDimension::Symbol(symbol) => match symbols.get(symbol) {
                    Some(expected) if *expected != symbolic_actual => anyhow::bail!(
                        "workflow value '{name}' axis {axis} binds symbol '{symbol}' to \
                         {symbolic_actual}, but it was already {expected}"
                    ),
                    Some(_) => {}
                    None => {
                        symbols.insert(symbol.clone(), symbolic_actual);
                    }
                },
                TensorDimension::Fixed(_) => {}
            }
        }
    }
    Ok(())
}

fn validate_state_recurrence(
    cell: &str,
    current: &Value,
    next: &Value,
    state: &onnx_genai_metadata::WorkflowStateCell,
    values: &PipelineTensors,
) -> anyhow::Result<()> {
    if current.dtype() != next.dtype() || current.shape().len() != next.shape().len() {
        anyhow::bail!("workflow state '{cell}' update must preserve dtype and rank");
    }
    match &state.recurrence {
        onnx_genai_metadata::ShapeRecurrence::Invariant => {
            if current.shape() != next.shape() {
                anyhow::bail!(
                    "workflow state '{cell}' is invariant but changed shape from {:?} to {:?}",
                    current.shape(),
                    next.shape()
                );
            }
        }
        onnx_genai_metadata::ShapeRecurrence::Growing {
            axis,
            increment,
            max,
        } => {
            for (index, (before, after)) in current.shape().iter().zip(next.shape()).enumerate() {
                if index != *axis && before != after {
                    anyhow::bail!(
                        "workflow state '{cell}' changed non-growing axis {index} from {before} \
                         to {after}"
                    );
                }
            }
            let growth = workflow_usize_rows(values, increment)?;
            let limits = workflow_usize_rows(values, max)?;
            let before = *current.shape().get(*axis).with_context(|| {
                format!("workflow state '{cell}' grows outside its tensor rank")
            })?;
            let after = *next.shape().get(*axis).with_context(|| {
                format!("workflow state '{cell}' grows outside its tensor rank")
            })?;
            if growth.len() > 1 || limits.len() > 1 {
                anyhow::ensure!(
                    state.service_group.is_some(),
                    "workflow state '{cell}' uses per-row growth without a KV service group"
                );
                let rows = growth.len().max(limits.len());
                for row in 0..rows {
                    let growth = growth[if growth.len() == 1 { 0 } else { row }];
                    let limit = limits[if limits.len() == 1 { 0 } else { row }];
                    anyhow::ensure!(
                        growth <= limit,
                        "workflow state '{cell}' row {row} growth {growth} exceeds maximum {limit}"
                    );
                }
                let storage_limit = limits.iter().copied().max().unwrap_or_default();
                anyhow::ensure!(
                    usize::try_from(after).is_ok_and(|after| after <= storage_limit),
                    "workflow state '{cell}' dense storage extent {after} exceeds maximum \
                     {storage_limit}"
                );
            } else {
                let growth =
                    i64::try_from(growth[0]).context("workflow state growth exceeds i64")?;
                let limit = i64::try_from(limits[0]).context("workflow state limit exceeds i64")?;
                let expected = before
                    .checked_add(growth)
                    .with_context(|| format!("workflow state '{cell}' shape growth overflowed"))?;
                if after != expected {
                    anyhow::bail!(
                        "workflow state '{cell}' growing axis {axis} changed from {before} to \
                         {after}, expected {expected}"
                    );
                }
                if after > limit {
                    anyhow::bail!(
                        "workflow state '{cell}' growing axis {axis} reached {after}, above maximum \
                         {limit}"
                    );
                }
            }
        }
        onnx_genai_metadata::ShapeRecurrence::Bounded { axis, max } => {
            for (index, (before, after)) in current.shape().iter().zip(next.shape()).enumerate() {
                if index != *axis && before != after {
                    anyhow::bail!(
                        "workflow state '{cell}' changed non-bounded axis {index} from {before} \
                         to {after}"
                    );
                }
            }
            let limit = i64::try_from(workflow_scalar_usize(values, max)?)
                .context("workflow state bounded-axis limit exceeds i64")?;
            let before = *current.shape().get(*axis).with_context(|| {
                format!("workflow state '{cell}' bounded axis is outside its tensor rank")
            })?;
            let after = *next.shape().get(*axis).with_context(|| {
                format!("workflow state '{cell}' bounded axis is outside its tensor rank")
            })?;
            if before > limit || after > limit {
                anyhow::bail!(
                    "workflow state '{cell}' bounded axis {axis} changed from {before} to {after}, \
                     above maximum {limit}"
                );
            }
        }
    }
    Ok(())
}

fn clone_pipeline_tensors(values: &PipelineTensors) -> anyhow::Result<PipelineTensors> {
    values
        .iter()
        .map(|(name, value)| Ok((name.clone(), clone_value(value)?)))
        .collect()
}

fn workflow_emitted_outputs(node: &WorkflowNode) -> std::collections::HashSet<String> {
    fn collect(node: &WorkflowNode, outputs: &mut std::collections::HashSet<String>) {
        match node {
            WorkflowNode::Sequence { nodes } => {
                for node in nodes {
                    collect(node, outputs);
                }
            }
            WorkflowNode::Loop { setup, body, .. } => {
                collect(setup, outputs);
                collect(body, outputs);
            }
            WorkflowNode::Branch { cases, default, .. } => {
                for case in cases.values() {
                    collect(case, outputs);
                }
                if let Some(default) = default {
                    collect(default, outputs);
                }
            }
            WorkflowNode::Emit { output, .. } => {
                outputs.insert(output.clone());
            }
            WorkflowNode::Invoke { .. }
            | WorkflowNode::Transfer { .. }
            | WorkflowNode::ExecutionIsland { .. } => {}
        }
    }
    let mut outputs = std::collections::HashSet::new();
    collect(node, &mut outputs);
    outputs
}

/// Outputs the workflow fills one request row at a time.
///
/// This is derived from structure, never from serialized row identities. An
/// emit is ragged when it carries a per-row `valid_length` or a per-row guard:
/// the rows then contribute different amounts and cannot share one dense
/// tensor. Raggedness is a property of the *output*, not of the individual
/// emit, so an output that is ragged anywhere is row-wise everywhere; that is
/// what lets an append loop mix a ragged accept step with a single-token
/// forced step and still produce one row per request.
pub(super) fn workflow_row_wise_outputs(node: &WorkflowNode) -> std::collections::HashSet<String> {
    fn collect(node: &WorkflowNode, outputs: &mut std::collections::HashSet<String>) {
        match node {
            WorkflowNode::Sequence { nodes } => {
                for node in nodes {
                    collect(node, outputs);
                }
            }
            WorkflowNode::Loop { setup, body, .. } => {
                collect(setup, outputs);
                collect(body, outputs);
            }
            WorkflowNode::Branch { cases, default, .. } => {
                for case in cases.values() {
                    collect(case, outputs);
                }
                if let Some(default) = default {
                    collect(default, outputs);
                }
            }
            WorkflowNode::Emit {
                output,
                valid_length,
                when,
                ..
            } => {
                if valid_length.is_some() || when.is_some() {
                    outputs.insert(output.clone());
                }
            }
            WorkflowNode::Invoke { .. }
            | WorkflowNode::Transfer { .. }
            | WorkflowNode::ExecutionIsland { .. } => {}
        }
    }
    let mut outputs = std::collections::HashSet::new();
    collect(node, &mut outputs);
    outputs
}

pub(super) fn compile_movable_emit_values(
    node: &WorkflowNode,
    workflow: &WorkflowSpec,
) -> std::collections::HashSet<String> {
    fn collect_uses(node: &WorkflowNode, uses: &mut HashMap<String, usize>) {
        let mut used = |value: &str| *uses.entry(value.to_string()).or_default() += 1;
        match node {
            WorkflowNode::Sequence { nodes } => {
                for node in nodes {
                    collect_uses(node, uses);
                }
            }
            WorkflowNode::Invoke { inputs, .. } => {
                for value in inputs.values() {
                    used(value);
                }
            }
            WorkflowNode::Loop {
                setup,
                body,
                continue_when,
                max_iterations,
                carried,
                ..
            } => {
                used(continue_when);
                used(max_iterations);
                for carry in carried {
                    used(&carry.current);
                    used(&carry.body_output);
                }
                collect_uses(setup, uses);
                collect_uses(body, uses);
            }
            WorkflowNode::Branch {
                predicate,
                cases,
                default,
                outputs,
                ..
            } => {
                used(predicate);
                for phi in outputs.values() {
                    for value in phi.cases.values() {
                        used(value);
                    }
                    if let Some(value) = &phi.default {
                        used(value);
                    }
                }
                for case in cases.values() {
                    collect_uses(case, uses);
                }
                if let Some(default) = default {
                    collect_uses(default, uses);
                }
            }
            WorkflowNode::Emit {
                value,
                when,
                valid_length,
                ..
            } => {
                used(value);
                if let Some(value) = when {
                    used(value);
                }
                if let Some(value) = valid_length {
                    used(value);
                }
            }
            WorkflowNode::Transfer { input, .. } => used(input),
            WorkflowNode::ExecutionIsland { .. } => {}
        }
    }

    fn collect_emits(node: &WorkflowNode, emits: &mut std::collections::HashSet<String>) {
        match node {
            WorkflowNode::Sequence { nodes } => {
                for node in nodes {
                    collect_emits(node, emits);
                }
            }
            WorkflowNode::Loop { setup, body, .. } => {
                collect_emits(setup, emits);
                collect_emits(body, emits);
            }
            WorkflowNode::Branch { cases, default, .. } => {
                for case in cases.values() {
                    collect_emits(case, emits);
                }
                if let Some(default) = default {
                    collect_emits(default, emits);
                }
            }
            WorkflowNode::Emit {
                value,
                when: None,
                valid_length: None,
                mode: WorkflowEmitMode::Replace,
                ..
            } => {
                emits.insert(value.clone());
            }
            WorkflowNode::Invoke { .. }
            | WorkflowNode::Emit { .. }
            | WorkflowNode::Transfer { .. }
            | WorkflowNode::ExecutionIsland { .. } => {}
        }
    }

    let mut uses = HashMap::new();
    collect_uses(node, &mut uses);
    let mut emits = std::collections::HashSet::new();
    collect_emits(node, &mut emits);
    emits.retain(|value| uses.get(value) == Some(&1) && !workflow.inputs.contains_key(value));
    emits
}

pub(super) fn compile_aliasable_output_values(
    node: &WorkflowNode,
) -> std::collections::HashSet<String> {
    fn collect_single_run_outputs(
        node: &WorkflowNode,
        repeated: bool,
        outputs: &mut std::collections::HashSet<String>,
    ) {
        match node {
            WorkflowNode::Sequence { nodes } => {
                for node in nodes {
                    collect_single_run_outputs(node, repeated, outputs);
                }
            }
            WorkflowNode::Invoke {
                outputs: produced, ..
            } if !repeated => outputs.extend(produced.values().cloned()),
            WorkflowNode::Loop { setup, body, .. } => {
                collect_single_run_outputs(setup, repeated, outputs);
                collect_single_run_outputs(body, true, outputs);
            }
            WorkflowNode::Branch { cases, default, .. } => {
                for case in cases.values() {
                    collect_single_run_outputs(case, repeated, outputs);
                }
                if let Some(default) = default {
                    collect_single_run_outputs(default, repeated, outputs);
                }
            }
            WorkflowNode::Invoke { .. }
            | WorkflowNode::Emit { .. }
            | WorkflowNode::Transfer { .. }
            | WorkflowNode::ExecutionIsland { .. } => {}
        }
    }

    let mut outputs = std::collections::HashSet::new();
    collect_single_run_outputs(node, false, &mut outputs);
    outputs
}

fn append_workflow_value(previous: &Value, next: &Value, axis: usize) -> anyhow::Result<Value> {
    if previous.dtype() != next.dtype() || previous.shape().len() != next.shape().len() {
        anyhow::bail!("workflow append emit requires matching dtype and rank");
    }
    let mut shape = previous.shape().to_vec();
    anyhow::ensure!(
        axis < shape.len(),
        "workflow append emit axis {axis} exceeds rank {}",
        shape.len()
    );
    for (index, (left, right)) in previous.shape().iter().zip(next.shape()).enumerate() {
        if index != axis && left != right {
            anyhow::bail!("workflow append emit requires equal non-appended dimensions");
        }
    }
    // Elements per slice of the appended axis, so a non-final axis interleaves
    // correctly instead of concatenating the wrong dimension.
    let inner = shape[axis + 1..]
        .iter()
        .map(|dimension| *dimension as usize)
        .product::<usize>();
    let left_width = shape[axis] as usize * inner;
    let right_width = next.shape()[axis] as usize * inner;
    let outer = shape[..axis]
        .iter()
        .map(|dimension| *dimension as usize)
        .product::<usize>();
    shape[axis] += next.shape()[axis];
    let dtype = previous.dtype();
    let element_size = dtype.size_of();
    let left = previous.to_raw_bytes()?;
    let right = next.to_raw_bytes()?;
    let mut data = Vec::with_capacity(left.len() + right.len());
    for row in 0..outer {
        data.extend_from_slice(
            &left[row * left_width * element_size..(row + 1) * left_width * element_size],
        );
        data.extend_from_slice(
            &right[row * right_width * element_size..(row + 1) * right_width * element_size],
        );
    }
    Value::from_raw_bytes(data, &shape, dtype).map_err(Into::into)
}

/// Axis an append emit grows along: the declared one, or the final axis.
fn append_axis(axis: Option<usize>, value: &Value) -> usize {
    axis.unwrap_or_else(|| value.shape().len().saturating_sub(1))
}

fn emit_chunk_contract(
    output: &onnx_genai_metadata::TensorContract,
    emitted: &Value,
    axis: Option<usize>,
) -> anyhow::Result<onnx_genai_metadata::TensorContract> {
    let mut contract = output.clone();
    if let Some(shape) = &mut contract.shape {
        let index = axis.unwrap_or(shape.len().saturating_sub(1));
        anyhow::ensure!(
            index < shape.len() && index < emitted.shape().len(),
            "workflow prefix/append emit axis {index} exceeds the value rank"
        );
        shape[index] = TensorDimension::Fixed(emitted.shape()[index]);
    }
    Ok(contract)
}

fn slice_workflow_prefix(
    value: &Value,
    valid_length: usize,
    axis: Option<usize>,
) -> anyhow::Result<Value> {
    let mut shape = value.shape().to_vec();
    let rank = shape.len();
    anyhow::ensure!(
        rank >= 1,
        "workflow emit valid_length requires a value with rank >= 1"
    );
    let index = axis.unwrap_or(rank - 1);
    anyhow::ensure!(
        index < rank,
        "workflow emit valid_length axis {index} exceeds rank {rank}"
    );
    let available =
        usize::try_from(shape[index]).context("workflow emit has a negative growth axis")?;
    if valid_length > available {
        anyhow::bail!(
            "workflow emit valid_length {valid_length} exceeds growth-axis extent {available}"
        );
    }
    let inner = shape[index + 1..]
        .iter()
        .map(|dimension| usize::try_from(*dimension))
        .collect::<Result<Vec<_>, _>>()
        .context("workflow emit has a negative dimension")?
        .into_iter()
        .product::<usize>();
    let outer = shape[..index]
        .iter()
        .map(|dimension| usize::try_from(*dimension))
        .collect::<Result<Vec<_>, _>>()
        .context("workflow emit has a negative dimension")?
        .into_iter()
        .product::<usize>();
    let dtype = value.dtype();
    let element_size = dtype.size_of();
    let source = value.to_raw_bytes()?;
    let kept = valid_length * inner;
    let stride = available * inner;
    let mut data = Vec::with_capacity(outer * kept * element_size);
    for row in 0..outer {
        let start = row * stride * element_size;
        data.extend_from_slice(&source[start..start + kept * element_size]);
    }
    shape[index] = i64::try_from(valid_length).context("workflow emit valid_length exceeds i64")?;
    Value::from_raw_bytes(data, &shape, dtype).map_err(Into::into)
}

#[allow(clippy::too_many_arguments)]
fn emit_workflow_rows(
    values: &mut PipelineTensors,
    tensor: &Value,
    value_name: &str,
    output: &str,
    output_contract: &onnx_genai_metadata::TensorContract,
    mode: &WorkflowEmitMode,
    guards: Option<&[bool]>,
    lengths: Option<&[usize]>,
    axis: Option<usize>,
    emit_counts: &mut HashMap<String, usize>,
    telemetry: &mut WorkflowRunTelemetry,
    symbols: &HashMap<String, i64>,
    dynamic_symbols: &std::collections::HashSet<String>,
) -> anyhow::Result<()> {
    let rows = tensor
        .shape()
        .first()
        .copied()
        .context("per-row workflow emit requires rank >= 1")?;
    let rows = usize::try_from(rows).context("workflow emit has a negative batch dimension")?;
    for cardinality in [guards.map(<[bool]>::len), lengths.map(<[usize]>::len)]
        .into_iter()
        .flatten()
    {
        anyhow::ensure!(
            cardinality == 1 || cardinality == rows,
            "workflow emit row control has {cardinality} values for batch {rows}"
        );
    }
    let mut row_names = vec![String::new(); rows];
    for row in 0..rows {
        let active = guards.is_none_or(|values| values[if values.len() == 1 { 0 } else { row }]);
        if !active {
            continue;
        }
        let length = lengths.map(|values| values[if values.len() == 1 { 0 } else { row }]);
        let emitted = slice_workflow_row(tensor, row, length, axis)?;
        let validation_contract = if length.is_some() || matches!(mode, WorkflowEmitMode::Append) {
            emit_chunk_contract(output_contract, &emitted, axis)?
        } else {
            output_contract.clone()
        };
        let mut row_symbols = symbols.clone();
        if let Some(TensorDimension::Symbol(batch)) = validation_contract
            .shape
            .as_ref()
            .and_then(|shape| shape.first())
        {
            row_symbols.insert(batch.clone(), 1);
        }
        validate_workflow_value(
            value_name,
            &emitted,
            &validation_contract,
            &mut row_symbols,
            dynamic_symbols,
        )?;
        if telemetry.first_emit_ns.is_none() {
            telemetry.first_emit_ns = telemetry
                .started
                .map(|started| started.elapsed().as_nanos());
        }
        telemetry.emit_events += 1;
        telemetry.emitted_elements += emitted.numel() as u64;
        let row_output = format!("{output}.row.{row}");
        row_names[row] = row_output.clone();
        match mode {
            WorkflowEmitMode::Replace => {
                values.insert(row_output, emitted);
            }
            WorkflowEmitMode::Append => {
                let appended = if let Some(previous) = values.get(&row_output) {
                    append_workflow_value(previous, &emitted, append_axis(axis, &emitted))?
                } else {
                    emitted
                };
                values.insert(row_output, appended);
            }
            WorkflowEmitMode::Event => {
                let index = emit_counts.entry(row_output.clone()).or_default();
                values.insert(format!("{row_output}.{index}"), clone_value(&emitted)?);
                *index += 1;
                values.insert(row_output, emitted);
            }
        }
    }
    // Row names are positional, so an inactive row keeps its slot and the
    // runtime can still map result row i onto its own request table.
    let entry = telemetry
        .row_outputs
        .entry(output.to_string())
        .or_insert_with(|| vec![String::new(); rows]);
    if entry.len() < rows {
        entry.resize(rows, String::new());
    }
    for (row, name) in row_names.into_iter().enumerate() {
        if !name.is_empty() {
            entry[row] = name;
        }
    }
    Ok(())
}

fn slice_workflow_row(
    value: &Value,
    row: usize,
    valid_length: Option<usize>,
    axis: Option<usize>,
) -> anyhow::Result<Value> {
    let mut shape = value.shape().to_vec();
    let rows = usize::try_from(
        *shape
            .first()
            .context("per-row workflow emit requires rank >= 1")?,
    )
    .context("workflow emit has a negative batch dimension")?;
    anyhow::ensure!(row < rows, "workflow emit row {row} exceeds batch {rows}");
    let row_elements = value.numel() / rows;
    let element_size = value.dtype().size_of();
    let source = value.to_raw_bytes()?;
    let start = row * row_elements * element_size;
    let row_value = Value::from_raw_bytes(
        source[start..start + row_elements * element_size].to_vec(),
        &{
            shape[0] = 1;
            shape
        },
        value.dtype(),
    )?;
    match valid_length {
        Some(length) => slice_workflow_prefix(&row_value, length, axis),
        None => Ok(row_value),
    }
}

fn merge_inactive_rows(
    current: &Value,
    next: &Value,
    active: &[bool],
    state: &onnx_genai_metadata::WorkflowStateCell,
) -> anyhow::Result<Value> {
    if active.len() == 1 || active.iter().all(|active| *active) {
        return clone_value(next);
    }
    anyhow::ensure!(current.dtype() == next.dtype());
    anyhow::ensure!(current.shape().len() == next.shape().len());
    let batch_axis = state
        .contract
        .batch_layout
        .request_axis()
        .context("mixed active row carry must be request-aligned")?;
    let rows = usize::try_from(
        *current
            .shape()
            .get(batch_axis)
            .context("mixed active row carry batch axis exceeds tensor rank")?,
    )?;
    let expansion = state.contract.batch_layout.request_expansion_factor();
    anyhow::ensure!(
        expansion > 0,
        "mixed active row carry has a zero request expansion factor"
    );
    anyhow::ensure!(
        active.len().saturating_mul(expansion) == rows,
        "loop active mask has {} requests with expansion {expansion} for state batch {rows}",
        active.len(),
    );
    anyhow::ensure!(
        next.shape().get(batch_axis) == current.shape().get(batch_axis),
        "mixed active row carry changed its request-aligned axis"
    );
    for (axis, (before, after)) in current.shape().iter().zip(next.shape()).enumerate() {
        if before != after {
            anyhow::ensure!(
                matches!(
                    &state.recurrence,
                    onnx_genai_metadata::ShapeRecurrence::Growing {
                        axis: growing_axis,
                        ..
                    } if *growing_axis == axis && after > before
                ),
                "mixed active rows require equal dense shapes or declared monotonic growth"
            );
        }
    }

    let current_shape = current
        .shape()
        .iter()
        .map(|dimension| usize::try_from(*dimension))
        .collect::<Result<Vec<_>, _>>()?;
    let next_shape = next
        .shape()
        .iter()
        .map(|dimension| usize::try_from(*dimension))
        .collect::<Result<Vec<_>, _>>()?;
    let dense_strides = |shape: &[usize]| -> anyhow::Result<Vec<usize>> {
        let mut strides = vec![1usize; shape.len()];
        for axis in (0..shape.len().saturating_sub(1)).rev() {
            strides[axis] = strides[axis + 1]
                .checked_mul(shape[axis + 1])
                .context("workflow carry stride overflowed")?;
        }
        Ok(strides)
    };
    let current_strides = dense_strides(&current_shape)?;
    let next_strides = dense_strides(&next_shape)?;
    let element_size = current.dtype().size_of();
    let current_bytes = current.to_raw_bytes()?;
    // A dense tensor has one physical extent for the whole batch. Declared
    // growth therefore allocates the new extent for inactive rows too: retain
    // the producer's new suffix, but restore every coordinate that existed in
    // the inactive row's prior semantic state.
    let mut merged = next.to_raw_bytes()?;
    for current_index in 0..current.numel() {
        let row = (current_index / current_strides[batch_axis]) % current_shape[batch_axis];
        if active[row / expansion] {
            continue;
        }
        let mut remainder = current_index;
        let mut next_index = 0usize;
        for axis in 0..current_shape.len() {
            let coordinate = remainder / current_strides[axis];
            remainder %= current_strides[axis];
            next_index = next_index
                .checked_add(
                    coordinate
                        .checked_mul(next_strides[axis])
                        .context("workflow carry index overflowed")?,
                )
                .context("workflow carry index overflowed")?;
        }
        let current_offset = current_index
            .checked_mul(element_size)
            .context("workflow carry byte offset overflowed")?;
        let next_offset = next_index
            .checked_mul(element_size)
            .context("workflow carry byte offset overflowed")?;
        let current_end = current_offset
            .checked_add(element_size)
            .context("workflow carry byte range overflowed")?;
        let next_end = next_offset
            .checked_add(element_size)
            .context("workflow carry byte range overflowed")?;
        merged[next_offset..next_end].copy_from_slice(&current_bytes[current_offset..current_end]);
    }
    Value::from_raw_bytes(merged, next.shape(), next.dtype()).map_err(Into::into)
}

fn share_workflow_value(values: &mut PipelineTensors, name: &str) -> anyhow::Result<Value> {
    if let Some(alias) = values.get(name).and_then(Value::try_alias_clone) {
        return alias.with_context(|| format!("workflow value '{name}' cannot retain its alias"));
    }
    let owner = values
        .remove(name)
        .with_context(|| format!("workflow value '{name}' is unavailable"))?;
    let shape = owner.shape().to_vec();
    let retained = Value::into_alias_with_shape(owner, &shape)?;
    let shared = retained
        .try_alias_clone()
        .context("new workflow value alias must retain its shared owner")??;
    values.insert(name.to_string(), retained);
    Ok(shared)
}

fn workflow_active_rows_without_inspection(
    values: &PipelineTensors,
    name: &str,
) -> anyhow::Result<Vec<bool>> {
    let value = values
        .get(name)
        .with_context(|| format!("workflow predicate value '{name}' is unavailable"))?;
    anyhow::ensure!(
        value.dtype() == DataType::Bool,
        "workflow predicate '{name}' must have bool dtype"
    );
    anyhow::ensure!(
        value.shape().len() <= 1,
        "workflow predicate '{name}' must be a scalar or rank-one row tensor"
    );
    let rows = value
        .shape()
        .first()
        .copied()
        .map(usize::try_from)
        .transpose()?
        .unwrap_or(1);
    anyhow::ensure!(rows > 0, "workflow predicate '{name}' must not be empty");
    Ok(vec![true; rows])
}

fn workflow_bool_rows(values: &PipelineTensors, name: &str) -> anyhow::Result<Vec<bool>> {
    let value = values
        .get(name)
        .with_context(|| format!("workflow predicate value '{name}' is unavailable"))?;
    anyhow::ensure!(
        value.dtype() == DataType::Bool,
        "workflow predicate '{name}' must have bool dtype"
    );
    let data = value.to_raw_bytes()?;
    anyhow::ensure!(
        !data.is_empty() && value.shape().len() <= 1,
        "workflow predicate '{name}' must be a scalar or rank-one row tensor"
    );
    Ok(data.into_iter().map(|value| value != 0).collect())
}

fn workflow_usize_rows(values: &PipelineTensors, name: &str) -> anyhow::Result<Vec<usize>> {
    let value = values
        .get(name)
        .with_context(|| format!("workflow integer control '{name}' is unavailable"))?;
    anyhow::ensure!(
        value.shape().len() <= 1,
        "workflow integer control '{name}' must be scalar or rank one"
    );
    let bytes = value.to_raw_bytes()?;
    let width = value.dtype().size_of();
    anyhow::ensure!(
        width > 0 && !bytes.is_empty() && bytes.len() % width == 0,
        "workflow integer control '{name}' must contain at least one value"
    );
    (0..bytes.len() / width)
        .map(|index| {
            let start = index * width;
            let mut one = PipelineTensors::new();
            one.insert(
                name.to_string(),
                Value::from_raw_bytes(bytes[start..start + width].to_vec(), &[], value.dtype())?,
            );
            workflow_scalar_usize(&one, name)
        })
        .collect()
}

/// Read the runtime-minted row selection, if the workflow declares one.
///
/// The selection is a gather of source positions within the current batch. It
/// is minted by the runtime scheduler each step and is deliberately opaque:
/// nothing in it identifies a request, a slot, or an epoch.
fn workflow_row_selection(
    workflow: &onnx_genai_metadata::WorkflowSpec,
    values: &PipelineTensors,
) -> anyhow::Result<Option<Vec<usize>>> {
    let Some(name) = workflow
        .inputs
        .iter()
        .find(|(_, input)| {
            matches!(
                &input.role,
                onnx_genai_metadata::SemanticInputRole::Runtime { role, .. }
                    if *role == RuntimeInputRole::RowSelection
            )
        })
        .map(|(name, _)| name.as_str())
    else {
        return Ok(None);
    };
    let Some(value) = values.get(name) else {
        return Ok(None);
    };
    let rows = value
        .to_vec_i64()
        .with_context(|| format!("row selection input '{name}' must be host int64"))?;
    rows.into_iter()
        .map(|row| {
            usize::try_from(row).with_context(|| {
                format!("row selection input '{name}' contains negative source row {row}")
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()
        .map(Some)
}

/// Component invocations reachable from a node, in order.
///
/// An execution island keeps its pre-fusion node as a fallback, which is the
/// only place the original per-component port bindings survive; fusion rewrites
/// them into one flat session signature.
fn invoke_bindings(node: &WorkflowNode) -> Vec<(&str, &BTreeMap<String, String>)> {
    match node {
        WorkflowNode::Invoke {
            component, inputs, ..
        } => vec![(component.as_str(), inputs)],
        WorkflowNode::Sequence { nodes } => nodes.iter().flat_map(invoke_bindings).collect(),
        _ => Vec::new(),
    }
}

fn workflow_scalar_usize(values: &PipelineTensors, name: &str) -> anyhow::Result<usize> {
    let value = values
        .get(name)
        .with_context(|| format!("workflow scalar value '{name}' is unavailable"))?;
    if value.shape().iter().try_fold(1usize, |size, dimension| {
        usize::try_from(*dimension)
            .ok()
            .and_then(|dimension| size.checked_mul(dimension))
    }) != Some(1)
    {
        anyhow::bail!("workflow scalar '{name}' must contain exactly one value");
    }
    let data = value.to_raw_bytes()?;
    let signed = |value: i128| {
        usize::try_from(value)
            .with_context(|| format!("workflow scalar '{name}' must be non-negative"))
    };
    let unsigned = |value: u128| {
        usize::try_from(value).with_context(|| format!("workflow scalar '{name}' exceeds usize"))
    };
    match value.dtype() {
        DataType::Int8 => signed(i8::from_ne_bytes([data[0]]) as i128),
        DataType::Int16 => signed(i16::from_ne_bytes(data[..2].try_into()?) as i128),
        DataType::Int32 => signed(i32::from_ne_bytes(data[..4].try_into()?) as i128),
        DataType::Int64 => signed(i64::from_ne_bytes(data[..8].try_into()?) as i128),
        DataType::Uint8 => unsigned(u8::from_ne_bytes([data[0]]) as u128),
        DataType::Uint16 => unsigned(u16::from_ne_bytes(data[..2].try_into()?) as u128),
        DataType::Uint32 => unsigned(u32::from_ne_bytes(data[..4].try_into()?) as u128),
        DataType::Uint64 => unsigned(u64::from_ne_bytes(data[..8].try_into()?) as u128),
        dtype => {
            anyhow::bail!("workflow scalar '{name}' must have an integer dtype, got {dtype:?}")
        }
    }
}

fn workflow_scalar_bool(values: &PipelineTensors, name: &str) -> anyhow::Result<bool> {
    let value = values
        .get(name)
        .with_context(|| format!("workflow predicate value '{name}' is unavailable"))?;
    match value.dtype() {
        DataType::Bool => {
            let data = value.to_raw_bytes()?;
            let [scalar] = data.as_slice() else {
                anyhow::bail!("workflow bool predicate '{name}' must contain exactly one value");
            };
            Ok(*scalar != 0)
        }
        _ => {
            let data = value.to_vec_i64()?;
            let [scalar] = data.as_slice() else {
                anyhow::bail!("workflow integer predicate '{name}' must contain exactly one value");
            };
            Ok(*scalar != 0)
        }
    }
}

fn workflow_scalar_key(values: &PipelineTensors, name: &str) -> anyhow::Result<String> {
    let value = values
        .get(name)
        .with_context(|| format!("workflow branch value '{name}' is unavailable"))?;
    if value.dtype() == DataType::Bool {
        return Ok(workflow_scalar_bool(values, name)?.to_string());
    }
    let data = value.to_vec_i64()?;
    let [scalar] = data.as_slice() else {
        anyhow::bail!("workflow branch tensor '{name}' must contain exactly one value");
    };
    Ok(scalar.to_string())
}

#[cfg(test)]
mod workflow_scalar_tests {
    use super::*;

    #[test]
    fn text_assembly_is_admitted_as_an_api_boundary_adapter() {
        assert!(supports_workflow_adapter("onnx-genai.text-assembly", "1"));
    }

    #[test]
    fn batched_loop_predicate_preserves_rows() {
        let mut values = PipelineTensors::new();
        values.insert(
            "done".to_string(),
            Value::from_raw_bytes(vec![0, 1], &[2], DataType::Bool).expect("bool tensor"),
        );

        assert_eq!(
            workflow_bool_rows(&values, "done").expect("batched predicate"),
            [false, true]
        );
    }

    #[test]
    fn batched_branch_key_is_not_silently_reduced() {
        let mut values = PipelineTensors::new();
        values.insert(
            "case".to_string(),
            Value::from_slice_i64(&[0, 1], &[2]).expect("integer tensor"),
        );

        let error = workflow_scalar_key(&values, "case").expect_err("batched branch key fails");
        assert!(error.to_string().contains("exactly one value"));
    }

    #[test]
    fn inactive_rows_keep_their_previous_carry() {
        let current = Value::from_slice_i64(&[1, 2, 3, 4], &[2, 2]).expect("current");
        let next = Value::from_slice_i64(&[10, 20, 30, 40], &[2, 2]).expect("next");
        let state: onnx_genai_metadata::WorkflowStateCell = serde_yaml::from_str(
            "contract: { dtype: int64, rank: 2, shape: [batch, 2], batch_layout: { kind: \
             request_aligned, axis: 0 } }\nscope: invocation\ninitializer: initial\nrecurrence: { \
             kind: invariant }",
        )
        .expect("state");
        let merged = merge_inactive_rows(&current, &next, &[true, false], &state)
            .expect("merge active rows");
        assert_eq!(merged.to_vec_i64().expect("merged values"), [10, 20, 3, 4]);
    }

    #[test]
    fn inactive_requests_keep_all_of_their_expanded_guidance_rows() {
        let current = Value::from_slice_i64(&[1, 2, 3, 4, 5, 6, 7, 8], &[4, 2]).expect("current");
        let next = Value::from_slice_i64(&[10, 20, 30, 40, 50, 60, 70, 80], &[4, 2]).expect("next");
        let state: onnx_genai_metadata::WorkflowStateCell = serde_yaml::from_str(
            "contract: { dtype: int64, rank: 2, shape: [guidance_rows, 2], batch_layout: { \
             kind: request_expanded, axis: 0, factor: 2 } }\nscope: invocation\ninitializer: \
             initial\nrecurrence: { kind: invariant }",
        )
        .expect("state");
        let merged = merge_inactive_rows(&current, &next, &[true, false], &state)
            .expect("merge expanded request rows");
        assert_eq!(
            merged.to_vec_i64().expect("merged values"),
            [10, 20, 30, 40, 5, 6, 7, 8]
        );
    }

    #[test]
    fn inactive_rows_keep_their_prefix_when_dense_state_grows() {
        let current = Value::from_slice_i64(&[1, 2, 3, 4], &[2, 2]).expect("current");
        let next = Value::from_slice_i64(&[10, 20, 30, 40, 50, 60], &[2, 3]).expect("next");
        let state: onnx_genai_metadata::WorkflowStateCell = serde_yaml::from_str(
            "contract: { dtype: int64, rank: 2, shape: [batch, context], batch_layout: { kind: \
             request_aligned, axis: 0 } }\nscope: invocation\ninitializer: initial\nrecurrence: { \
             kind: growing, axis: 1, increment: one, max: limit }",
        )
        .expect("state");
        let merged = merge_inactive_rows(&current, &next, &[true, false], &state)
            .expect("merge growing rows");
        assert_eq!(
            merged.to_vec_i64().expect("merged values"),
            [10, 20, 30, 3, 4, 60]
        );
    }

    #[test]
    fn inactive_rows_follow_the_declared_nonleading_batch_axis() {
        let current = Value::from_slice_i64(&[1, 2, 3, 4], &[2, 2]).expect("current");
        let next = Value::from_slice_i64(&[10, 20, 30, 40], &[2, 2]).expect("next");
        let state: onnx_genai_metadata::WorkflowStateCell = serde_yaml::from_str(
            "contract: { dtype: int64, rank: 2, shape: [features, batch], batch_layout: { kind: \
             request_aligned, axis: 1 } }\nscope: invocation\ninitializer: initial\nrecurrence: { \
             kind: invariant }",
        )
        .expect("state");
        let merged = merge_inactive_rows(&current, &next, &[true, false], &state)
            .expect("merge active rows");
        assert_eq!(merged.to_vec_i64().expect("merged values"), [10, 2, 30, 4]);
    }

    #[test]
    fn shared_loop_output_remains_available_for_multiple_carries() {
        let mut values = PipelineTensors::new();
        values.insert(
            "body.output".to_string(),
            Value::from_slice_i64(&[7, 8], &[2]).expect("body output"),
        );

        let first = share_workflow_value(&mut values, "body.output").expect("first carry alias");
        let second = share_workflow_value(&mut values, "body.output").expect("second carry alias");

        assert_eq!(first.to_vec_i64().expect("first values"), [7, 8]);
        assert_eq!(second.to_vec_i64().expect("second values"), [7, 8]);
        assert_eq!(
            values["body.output"]
                .to_vec_i64()
                .expect("retained output values"),
            [7, 8]
        );
    }

    #[test]
    fn row_emit_slices_each_runtime_prefix_and_suppresses_guarded_rows() {
        let tensor = Value::from_slice_i64(&[10, 11, 12, 20, 21, 22], &[2, 3]).expect("row tensor");
        let mut values = PipelineTensors::new();
        let mut counts = HashMap::new();
        let mut telemetry = WorkflowRunTelemetry::default();
        let contract: TensorContract =
            serde_yaml::from_str("{ dtype: int64, rank: 2, shape: [batch, sequence] }")
                .expect("contract");
        emit_workflow_rows(
            &mut values,
            &tensor,
            "token",
            "tokens",
            &contract,
            &WorkflowEmitMode::Replace,
            Some(&[true, false][..]),
            Some(&[2, 1][..]),
            None,
            &mut counts,
            &mut telemetry,
            &HashMap::new(),
            &std::collections::HashSet::new(),
        )
        .expect("row emit");

        assert_eq!(
            values["tokens.row.0"]
                .to_vec_i64()
                .expect("first row values"),
            [10, 11]
        );
        assert!(!values.contains_key("tokens.row.1"));
        assert_eq!(telemetry.emit_events, 1);
    }

    #[test]
    fn single_row_vector_control_still_uses_ragged_output_namespace() {
        let tensor = Value::from_slice_i64(&[10, 11, 12], &[1, 3]).expect("row tensor");
        let mut values = PipelineTensors::new();
        let mut counts = HashMap::new();
        let mut telemetry = WorkflowRunTelemetry::default();
        let contract: TensorContract =
            serde_yaml::from_str("{ dtype: int64, rank: 2, shape: [batch, sequence] }")
                .expect("contract");
        emit_workflow_rows(
            &mut values,
            &tensor,
            "token",
            "tokens",
            &contract,
            &WorkflowEmitMode::Replace,
            None,
            Some(&[2][..]),
            None,
            &mut counts,
            &mut telemetry,
            &HashMap::new(),
            &std::collections::HashSet::new(),
        )
        .expect("row emit");
        assert_eq!(
            values["tokens.row.0"].to_vec_i64().expect("row values"),
            [10, 11]
        );
        assert!(!values.contains_key("tokens"));
    }

    #[test]
    fn row_replace_without_ragged_length_preserves_declared_extent_validation() {
        let tensor = Value::from_slice_i64(&[10, 11, 12], &[1, 3]).expect("row tensor");
        let contract: TensorContract =
            serde_yaml::from_str("{ dtype: int64, rank: 2, shape: [batch, 4] }").expect("contract");
        let error = emit_workflow_rows(
            &mut PipelineTensors::new(),
            &tensor,
            "token",
            "tokens",
            &contract,
            &WorkflowEmitMode::Replace,
            Some(&[true][..]),
            None,
            None,
            &mut HashMap::new(),
            &mut WorkflowRunTelemetry::default(),
            &HashMap::new(),
            &std::collections::HashSet::new(),
        )
        .expect_err("fixed output extent must be checked");
        let message = error.to_string();
        assert!(message.contains("axis 1 is 3, expected 4"), "{message}");
    }

    #[test]
    fn row_emit_keeps_every_active_row_in_its_own_positional_slot() {
        let tensor = Value::from_slice_i64(&[10, 20], &[2, 1]).expect("row tensor");
        let contract: TensorContract =
            serde_yaml::from_str("{ dtype: int64, rank: 2, shape: [batch, sequence] }")
                .expect("contract");
        let mut values = PipelineTensors::new();
        let mut telemetry = WorkflowRunTelemetry::default();
        emit_workflow_rows(
            &mut values,
            &tensor,
            "token",
            "tokens",
            &contract,
            &WorkflowEmitMode::Append,
            Some(&[true, true]),
            Some(&[1, 1]),
            None,
            &mut HashMap::new(),
            &mut telemetry,
            &HashMap::new(),
            &std::collections::HashSet::new(),
        )
        .expect("row emit");
        // Two rows that would previously have collided on a duplicated identity
        // now occupy distinct positions, so no identity uniqueness check is
        // needed and none can be violated.
        assert_eq!(values["tokens.row.0"].to_vec_i64().expect("row 0"), [10]);
        assert_eq!(values["tokens.row.1"].to_vec_i64().expect("row 1"), [20]);
        assert_eq!(
            telemetry.row_outputs["tokens"],
            vec!["tokens.row.0".to_string(), "tokens.row.1".to_string()]
        );
    }

    #[test]
    fn true_zero_length_row_emits_an_empty_value() {
        let tensor = Value::from_slice_i64(&[10], &[1, 1]).expect("row tensor");
        let contract: TensorContract =
            serde_yaml::from_str("{ dtype: int64, rank: 2, shape: [batch, sequence] }")
                .expect("contract");
        let mut values = PipelineTensors::new();
        let mut telemetry = WorkflowRunTelemetry::default();
        emit_workflow_rows(
            &mut values,
            &tensor,
            "token",
            "tokens",
            &contract,
            &WorkflowEmitMode::Append,
            Some(&[true]),
            Some(&[0]),
            None,
            &mut HashMap::new(),
            &mut telemetry,
            &HashMap::new(),
            &std::collections::HashSet::new(),
        )
        .expect("zero-length emit");
        assert_eq!(values["tokens.row.0"].shape(), [1, 0]);
        assert_eq!(
            telemetry.row_outputs["tokens"].first().map(String::as_str),
            Some("tokens.row.0")
        );
    }

    #[test]
    fn growing_state_uses_recurrence_instead_of_freezing_its_symbol() {
        let contract: TensorContract =
            serde_yaml::from_str("{ dtype: int64, rank: 2, shape: [batch, sequence] }")
                .expect("contract");
        let mut symbols = HashMap::new();
        let dynamic_symbols = std::collections::HashSet::from(["sequence".to_string()]);
        let current = Value::from_slice_i64(&[1, 2], &[1, 2]).expect("current");
        let next = Value::from_slice_i64(&[1, 2, 3], &[1, 3]).expect("next");
        validate_workflow_value(
            "current",
            &current,
            &contract,
            &mut symbols,
            &dynamic_symbols,
        )
        .expect("current contract");
        validate_workflow_value("next", &next, &contract, &mut symbols, &dynamic_symbols)
            .expect("growing symbol remains dynamic");

        let state: onnx_genai_metadata::WorkflowStateCell = serde_yaml::from_str(
            r#"
contract: { dtype: int64, rank: 2, shape: [batch, sequence] }
scope: invocation
initializer: initial
recurrence: { kind: growing, axis: 1, increment: accepted, max: max_context }
"#,
        )
        .expect("state");
        let mut values = PipelineTensors::new();
        values.insert(
            "accepted".to_string(),
            Value::from_slice_i64(&[1], &[]).expect("increment"),
        );
        values.insert(
            "max_context".to_string(),
            Value::from_slice_i64(&[4], &[]).expect("limit"),
        );
        validate_state_recurrence("tokens", &current, &next, &state, &values)
            .expect("bounded growth validates");

        let invalid = Value::from_slice_i64(&[1, 2, 3, 4], &[1, 4]).expect("invalid next");
        let error = validate_state_recurrence("tokens", &current, &invalid, &state, &values)
            .expect_err("wrong increment fails");
        assert!(error.to_string().contains("expected 3"));
    }

    #[test]
    fn kv_service_accepts_per_row_growth_controls() {
        let state: onnx_genai_metadata::WorkflowStateCell = serde_yaml::from_str(
            r#"
contract: { dtype: int64, rank: 2, shape: [batch, sequence] }
scope: invocation
initializer: initial
recurrence: { kind: growing, axis: 1, increment: accepted, max: max_context }
service_group: decoder_cache
"#,
        )
        .expect("state");
        let current = Value::from_slice_i64(&[1, 2, 3, 4], &[2, 2]).expect("current");
        let next = Value::from_slice_i64(&[1, 2, 9, 3, 4, 0], &[2, 3]).expect("next");
        let mut values = PipelineTensors::new();
        values.insert(
            "accepted".to_string(),
            Value::from_slice_i64(&[1, 0], &[2]).expect("per-row growth"),
        );
        values.insert(
            "max_context".to_string(),
            Value::from_slice_i64(&[4], &[]).expect("limit"),
        );
        validate_state_recurrence("cache", &current, &next, &state, &values)
            .expect("KV service owns per-row logical lengths");
    }

    #[test]
    fn literals_and_append_support_declared_runtime_dtypes() {
        let int_contract: TensorContract =
            serde_yaml::from_str("{ dtype: int16, rank: 1, shape: [2] }").expect("contract");
        let integer = workflow_literal_value(
            &LiteralValue::Scalar(ScalarValue::Integer(7)),
            &int_contract,
        )
        .expect("int16 literal");
        assert_eq!(integer.dtype(), DataType::Int16);

        let half_contract: TensorContract =
            serde_yaml::from_str("{ dtype: float16, rank: 1, shape: [2] }").expect("contract");
        let left = workflow_literal_value(
            &LiteralValue::Scalar(ScalarValue::Float(1.0)),
            &half_contract,
        )
        .expect("half literal");
        let right = workflow_literal_value(
            &LiteralValue::Scalar(ScalarValue::Float(2.0)),
            &half_contract,
        )
        .expect("half literal");
        let appended = append_workflow_value(&left, &right, 0).expect("half append");
        assert_eq!(appended.dtype(), DataType::Float16);
        assert_eq!(appended.shape(), &[4]);
    }

    #[test]
    fn element_literals_carry_per_position_constants() {
        // Interleaved full-duplex models publish a per-stream delay pattern; it
        // is a tensor constant, not a broadcast scalar.
        let contract: TensorContract =
            serde_yaml::from_str("{ dtype: int64, rank: 1, shape: [5] }").expect("contract");
        let delays = LiteralValue::Elements(vec![
            ScalarValue::Integer(0),
            ScalarValue::Integer(0),
            ScalarValue::Integer(1),
            ScalarValue::Integer(1),
            ScalarValue::Integer(1),
        ]);
        let value = workflow_literal_value(&delays, &contract).expect("delay literal");
        assert_eq!(value.dtype(), DataType::Int64);
        assert_eq!(value.shape(), &[5]);
        assert_eq!(value.to_vec_i64().expect("elements"), vec![0, 0, 1, 1, 1]);

        let short = LiteralValue::Elements(vec![ScalarValue::Integer(0)]);
        let Err(error) = workflow_literal_value(&short, &contract) else {
            panic!("element count must be checked against the contract");
        };
        assert!(error.to_string().contains("elements"), "{error}");
    }

    #[test]
    fn append_interleaves_along_a_non_final_axis() {
        // Video frames grow along axis 2 of [batch, channels, frames, height],
        // so appending must interleave per (batch, channel) slice instead of
        // concatenating at the end of the buffer like a token sequence does.
        let previous =
            Value::from_slice_i64(&[1, 2, 3, 4], &[1, 2, 1, 2]).expect("published frames");
        let next = Value::from_slice_i64(&[5, 6, 7, 8], &[1, 2, 1, 2]).expect("next chunk");

        let appended = append_workflow_value(&previous, &next, 2).expect("temporal append");

        assert_eq!(appended.shape(), &[1, 2, 2, 2]);
        assert_eq!(
            appended.to_vec_i64().expect("appended frames"),
            // channel 0 keeps [1, 2] then [5, 6]; channel 1 keeps [3, 4] then [7, 8]
            [1, 2, 5, 6, 3, 4, 7, 8]
        );

        let last_axis = append_workflow_value(&previous, &next, 3).expect("trailing append");
        assert_eq!(last_axis.shape(), &[1, 2, 1, 4]);
        assert_eq!(
            last_axis.to_vec_i64().expect("appended row"),
            [1, 2, 5, 6, 3, 4, 7, 8]
        );

        assert!(
            append_workflow_value(&previous, &next, 4).is_err(),
            "axis must be within rank"
        );
    }
}

#[cfg(test)]
mod workflow_sampling_binding_tests {
    use super::*;
    use crate::{GenerateOptions, GeneratePrompt, GenerateRequest};
    use onnx_genai_metadata::BatchLayout;

    fn request(configure: impl FnOnce(&mut GenerateOptions)) -> GenerateRequest {
        let mut request = GenerateRequest::new("hello");
        configure(&mut request.options);
        request
    }

    #[test]
    fn greedy_requests_bind_argmax_forcing_sampler_parameters() {
        // The runtime's own defaults leave truncation disabled, so a greedy
        // request must not be forwarded verbatim to a graph sampler.
        let options = GenerateOptions::default();
        assert!(options.greedy, "runtime default is greedy");
        assert_eq!(options.top_k, 0, "top_k default disables truncation");
        assert_eq!(options.temperature, 1.0, "temperature default is neutral");

        let sampling = effective_sampling(&request(|options| {
            options.greedy = true;
            options.temperature = 0.7;
            options.top_k = 64;
            options.top_p = 0.9;
            options.min_p = 0.05;
        }));

        assert_eq!(sampling.temperature, 0.0);
        assert_eq!(sampling.top_k, 1);
        assert_eq!(sampling.top_p, 1.0);
        assert_eq!(sampling.min_p, 0.0);
    }

    #[test]
    fn sampling_requests_bind_caller_parameters_unchanged() {
        let sampling = effective_sampling(&request(|options| {
            options.greedy = false;
            options.temperature = 0.7;
            options.top_k = 64;
            options.top_p = 0.9;
            options.min_p = 0.05;
        }));

        assert_eq!(sampling.temperature, 0.7);
        assert_eq!(sampling.top_k, 64);
        assert_eq!(sampling.top_p, 0.9);
        assert_eq!(sampling.min_p, 0.05);
    }

    #[test]
    fn multi_row_prompt_tokens_bind_as_rank_two() {
        let mut request = request(|_| {});
        request.prompt = GeneratePrompt::TokenRows(vec![vec![1, 2, 3], vec![4, 5, 6]]);
        let contract = TensorContract {
            dtype: "int64".to_string(),
            rank: 2,
            shape: None,
            optional: false,
            batch_layout: BatchLayout::default(),
        };

        let value = workflow_request_value(&RuntimeInputRole::PromptTokens, &request, &contract)
            .expect("prompt binding")
            .expect("prompt value");
        assert_eq!(value.shape(), &[2, 3]);
        assert_eq!(
            value.to_vec_i64().expect("prompt rows"),
            vec![1, 2, 3, 4, 5, 6]
        );
    }

    #[test]
    fn multi_row_prompt_tokens_require_equal_lengths() {
        let mut request = request(|_| {});
        request.prompt = GeneratePrompt::TokenRows(vec![vec![1, 2], vec![3]]);
        let contract = TensorContract {
            dtype: "int64".to_string(),
            rank: 2,
            shape: None,
            optional: false,
            batch_layout: BatchLayout::default(),
        };

        let error =
            match workflow_request_value(&RuntimeInputRole::PromptTokens, &request, &contract) {
                Ok(_) => panic!("ragged rows must fail"),
                Err(error) => error,
            };
        assert!(error.to_string().contains("equal lengths"));
    }

    #[test]
    fn greedy_sampling_roles_resolve_to_argmax_forcing_values() {
        let request = request(|options| {
            options.greedy = true;
            options.top_k = 0;
        });
        let contract = TensorContract {
            dtype: "int64".to_string(),
            rank: 1,
            shape: Some(vec![TensorDimension::Fixed(1)]),
            optional: false,
            batch_layout: BatchLayout::default(),
        };

        let top_k = workflow_request_value(&RuntimeInputRole::SamplingTopK, &request, &contract)
            .expect("top_k binding")
            .expect("top_k value");
        assert_eq!(top_k.to_vec_i64().expect("i64 rows"), vec![1]);

        let temperature =
            workflow_request_value(&RuntimeInputRole::SamplingTemperature, &request, &contract)
                .expect("temperature binding")
                .expect("temperature value");
        assert_eq!(temperature.to_vec_f32().expect("f32 rows"), vec![0.0]);
    }
}

/// The node-host seam: a runtime executor runs, or the declared step fails.
#[cfg(test)]
mod node_host_tests {
    use super::*;

    /// A host that records what it was asked to run and defines its outputs by
    /// copying the first input, so a test can tell "the executor ran" from
    /// "the interpreter fell through to a port copy".
    struct RecordingHost {
        contract: String,
        seen: Vec<String>,
    }

    impl WorkflowNodeHost for RecordingHost {
        fn hosted_contracts(&self) -> &'static [&'static str] {
            super::super::generation::DECODE_CORE_CONTRACTS
        }

        fn execute_contract_node(
            &mut self,
            request: WorkflowNodeRequest<'_>,
        ) -> anyhow::Result<bool> {
            if request.contract != self.contract {
                return Ok(false);
            }
            self.seen.push(request.component.to_string());
            let source = request
                .inputs
                .values()
                .next()
                .context("the test host expects one input")?;
            let tensor = clone_value(
                request
                    .values
                    .get(source)
                    .context("the test host's input must be bound")?,
            )?;
            for output in request.outputs.values() {
                request.values.insert(output.clone(), clone_value(&tensor)?);
            }
            Ok(true)
        }
    }

    /// The registry is the single answer to "can this runtime run this step?".
    ///
    /// A contract absent from it is a package this build cannot execute, and
    /// the operator's fix is a version, not a retry. A contract present in it
    /// but unhandled at runtime is a missing resource for *this request*. The
    /// two errors say different things because they have different fixes.
    #[test]
    fn the_registry_names_exactly_the_contracts_this_runtime_implements() {
        let registry = runtime_contract_registry();
        assert!(
            registry
                .contains(onnx_genai_metadata::decoder_workflow::AUTOREGRESSIVE_DECODE_CONTRACT)
        );
        assert!(registry.contains(onnx_genai_metadata::decoder_workflow::TOKEN_POLICY_CONTRACT));
        assert!(
            !registry.contains("vendor.not-a-contract-we-implement"),
            "an unregistered contract must not be claimed; a package naming it is one this \
             build cannot run, and saying otherwise sends an operator looking for a bug that \
             is really a version mismatch"
        );
    }

    /// A binding whose contract no host implements is an error.
    ///
    /// The alternative — falling through to the same-name port copy — would
    /// define the outputs with the wrong tensors and let a declared step
    /// silently not happen, which is indistinguishable from success at the API.
    #[test]
    fn an_unimplemented_contract_fails_closed() {
        let mut host = RecordingHost {
            contract: "vendor.something-else".to_string(),
            seen: Vec::new(),
        };
        let request = WorkflowNodeRequest {
            contract: "onnx-genai.token-policy",
            component: "token_policy",
            inputs: &BTreeMap::new(),
            outputs: &BTreeMap::new(),
            values: &mut PipelineTensors::default(),
        };
        assert!(
            !host.execute_contract_node(request).expect("no error"),
            "a host must decline a contract it does not implement, so the interpreter can \
             report it rather than silently doing nothing"
        );
    }

    /// A host that claims a contract is the implementation of that node.
    #[test]
    fn a_host_executor_defines_the_declared_outputs() {
        let mut values = PipelineTensors::default();
        values.insert(
            "in".to_string(),
            Value::from_slice_i64(&[7], &[1]).expect("value"),
        );
        let inputs = BTreeMap::from([("logits".to_string(), "in".to_string())]);
        let outputs = BTreeMap::from([("token".to_string(), "out".to_string())]);
        let mut host = RecordingHost {
            contract: "onnx-genai.token-policy".to_string(),
            seen: Vec::new(),
        };

        let handled = host
            .execute_contract_node(WorkflowNodeRequest {
                contract: "onnx-genai.token-policy",
                component: "token_policy",
                inputs: &inputs,
                outputs: &outputs,
                values: &mut values,
            })
            .expect("the host runs");

        assert!(handled);
        assert_eq!(host.seen, vec!["token_policy".to_string()]);
        assert!(
            values.contains_key("out"),
            "the host must define every output the invocation declares"
        );
    }
}

/// The loop records why it stopped.
#[cfg(test)]
mod loop_stop_cause_tests {
    use super::*;

    /// A fresh run has not yet ended by predicate.
    ///
    /// The flag must start false, because it is read as "the model stopped
    /// itself" and a default of true would report every bound-exhausted run as
    /// an EOS — the exact inversion of the bug it fixes.
    #[test]
    fn the_stop_cause_defaults_to_the_bound() {
        let telemetry = WorkflowRunTelemetry::default();
        assert!(
            !telemetry.loop_ended_by_predicate,
            "an unstarted run must not claim the model ended it"
        );
    }

    /// The counters carry the cause out of the run, which is the only way the
    /// generate path can see it.
    #[test]
    fn the_stop_cause_reaches_the_counters() {
        let mut counters = WorkflowPerformanceCounters::default();
        let telemetry = WorkflowRunTelemetry {
            loop_ended_by_predicate: true,
            ..WorkflowRunTelemetry::default()
        };
        counters.last_loop_ended_by_predicate = telemetry.loop_ended_by_predicate;
        assert!(counters.last_loop_ended_by_predicate);
    }
}

/// The parts of a declared loop, borrowed from the compiled graph.
///
/// Named so the loop's phases can be driven by two callers without either
/// restating what a loop *is*: the interpreter runs them back to back, and the
/// scheduler's per-token drive runs one iteration per call, resuming where the
/// last one stopped. A second statement of the semantics is exactly the thing
/// that could disagree with this one.
pub(crate) struct WorkflowLoopPlan<'n> {
    pub(crate) setup: &'n WorkflowNode,
    pub(crate) body: &'n WorkflowNode,
    pub(crate) continue_when: &'n str,
    pub(crate) max_iterations: &'n str,
    pub(crate) termination: &'n onnx_genai_metadata::WorkflowLoopTermination,
    pub(crate) iteration: &'n Option<onnx_genai_metadata::WorkflowLoopIteration>,
    pub(crate) carried: &'n [onnx_genai_metadata::WorkflowLoopCarry],
}

/// Whether this loop body publishes the package's token stream.
///
/// The question is answered from the workflow: a loop is the generation loop
/// when it emits into an output the package declares with `role: tokens`. A
/// package with several loops has exactly one such loop, and a package with
/// none — an image pipeline — has no generation stop to report.
fn loop_emits_generation_tokens(body: &WorkflowNode, workflow: &WorkflowSpec) -> bool {
    workflow_emitted_outputs(body).iter().any(|output| {
        workflow
            .outputs
            .get(output)
            .is_some_and(|declared| declared.role == WorkflowOutputRole::Tokens)
    })
}

/// Release the loop's induction value once no iteration can read it.
pub(crate) fn end_workflow_loop(plan: &WorkflowLoopPlan<'_>, values: &mut PipelineTensors) {
    if let Some(iteration) = plan.iteration {
        values.remove(&iteration.value);
    }
}

impl WorkflowRuntime {
    /// Run a loop's setup and seed its carried state, returning the declared
    /// iteration bound.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn begin_workflow_loop(
        &self,
        plan: &WorkflowLoopPlan<'_>,
        workflow: &WorkflowSpec,
        values: &mut PipelineTensors,
        symbols: &mut HashMap<String, i64>,
        dynamic_symbols: &std::collections::HashSet<String>,
        emit_counts: &mut HashMap<String, usize>,
        final_state_refs: &mut HashMap<String, String>,
        component_overrides: &HashMap<String, String>,
        telemetry: &mut WorkflowRunTelemetry,
        host: &mut Option<&mut dyn WorkflowNodeHost>,
    ) -> anyhow::Result<usize> {
        let WorkflowLoopPlan {
            setup,
            max_iterations,
            iteration,
            carried,
            ..
        } = plan;
        let carried = *carried;
        self.run_workflow_node(
            setup,
            workflow,
            values,
            symbols,
            dynamic_symbols,
            emit_counts,
            final_state_refs,
            component_overrides,
            telemetry,
            host,
        )?;
        self.materialize_workflow_value(values, max_iterations)?;
        for carry in carried {
            let state = workflow.state.get(&carry.cell).with_context(|| {
                format!("workflow loop carries undeclared state '{}'", carry.cell)
            })?;
            let seeded = values
                .get(&session_state_value_name(&carry.cell))
                .or_else(|| values.get(&carry.current));
            // A runtime-managed cell whose seed the package never materializes
            // is the service's buffer, not an SSA value. Recording the
            // reference is what the interpreter owes it; inventing a tensor
            // would be a second answer about where that buffer lives.
            if seeded.is_none() && state.management == onnx_genai_metadata::StateManagement::Runtime
            {
                final_state_refs.insert(carry.cell.clone(), carry.next.clone());
                continue;
            }
            let initializer = seeded.with_context(|| {
                format!(
                    "workflow state '{}' loop initializer '{}' is unavailable after setup",
                    carry.cell, carry.current
                )
            })?;
            validate_workflow_value(
                &carry.current,
                initializer,
                &state.contract,
                symbols,
                dynamic_symbols,
            )?;
            let initial_value = clone_value(initializer)?;
            values.insert(carry.next.clone(), initial_value);
            final_state_refs.insert(carry.cell.clone(), carry.next.clone());
        }
        let limit = workflow_scalar_usize(values, max_iterations)?;
        if let Some(iteration) = iteration
            && values.contains_key(&iteration.value)
        {
            anyhow::bail!(
                "workflow loop iteration value '{}' shadows an existing SSA value",
                iteration.value
            );
        }
        Ok(limit)
    }

    /// Advance a loop by one iteration.
    ///
    /// Returns `false` when the liveness predicate ended it, which is the only
    /// stop the loop itself decides; exhausting the declared bound is the
    /// caller's business.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn advance_workflow_loop(
        &self,
        plan: &WorkflowLoopPlan<'_>,
        workflow: &WorkflowSpec,
        index: usize,
        values: &mut PipelineTensors,
        symbols: &mut HashMap<String, i64>,
        dynamic_symbols: &std::collections::HashSet<String>,
        emit_counts: &mut HashMap<String, usize>,
        final_state_refs: &mut HashMap<String, String>,
        component_overrides: &HashMap<String, String>,
        telemetry: &mut WorkflowRunTelemetry,
        host: &mut Option<&mut dyn WorkflowNodeHost>,
    ) -> anyhow::Result<bool> {
        let WorkflowLoopPlan {
            body,
            continue_when,
            termination,
            iteration,
            carried,
            ..
        } = plan;
        let carried = *carried;
        let emits_generation_tokens = loop_emits_generation_tokens(body, workflow);
        // Skipping the liveness read is an optimization for a predicate the
        // caller said it does not care about: with `stop_on_eos` off, an
        // in-graph EOS predicate cannot end the loop, so reading it every step
        // costs a device→host sync for an answer that is always true.
        //
        // A predicate a *runtime executor* writes is a different thing. It is
        // already on the host, and it carries more than EOS — the context
        // bound, stop strings, a constraint reaching its end. Skipping it there
        // would ignore stops the caller did ask for, so the optimization does
        // not apply when a host implements a node in this body.
        let max_iterations_only = telemetry.max_iterations_only
            && host.is_none()
            && **termination == onnx_genai_metadata::WorkflowLoopTermination::GenerationEos;
        let active_rows = if max_iterations_only {
            workflow_active_rows_without_inspection(values, continue_when)?
        } else {
            self.materialize_workflow_value(values, continue_when)?;
            workflow_bool_rows(values, continue_when)?
        };
        if !active_rows.iter().any(|active| *active) {
            // The loop stopped because the model said it was done, not because
            // it ran out of budget. Recording which is the only way a caller
            // can report the real reason.
            //
            // Recorded only for the loop that emits the token stream. A
            // workflow may declare several — a denoise loop beside a generation
            // loop, or an inner loop within one — and any of them ending by its
            // own predicate would otherwise latch this and make a generation
            // that exhausted its budget report EOS.
            if emits_generation_tokens {
                telemetry.loop_ended_by_predicate = true;
            }
            return Ok(false);
        }
        telemetry.loop_iterations += 1;
        if let Some(iteration) = iteration {
            values.insert(
                iteration.value.clone(),
                workflow_iteration_value(index, &iteration.contract, symbols)?,
            );
        }
        for carry in carried {
            if carry.body_input == carry.next {
                continue;
            }
            let current = values
                .get(&carry.next)
                .with_context(|| format!("workflow loop value '{}' is unavailable", carry.next))?;
            values.insert(carry.body_input.clone(), clone_value(current)?);
        }
        self.run_workflow_node(
            body,
            workflow,
            values,
            symbols,
            dynamic_symbols,
            emit_counts,
            final_state_refs,
            component_overrides,
            telemetry,
            host,
        )?;
        for carry in carried {
            let state = workflow.state.get(&carry.cell).with_context(|| {
                format!("workflow loop carries undeclared state '{}'", carry.cell)
            })?;
            match &state.recurrence {
                onnx_genai_metadata::ShapeRecurrence::Growing { increment, max, .. } => {
                    self.materialize_workflow_value(values, increment)?;
                    self.materialize_workflow_value(values, max)?;
                }
                onnx_genai_metadata::ShapeRecurrence::Bounded { max, .. } => {
                    self.materialize_workflow_value(values, max)?;
                }
                onnx_genai_metadata::ShapeRecurrence::Invariant => {}
            }
            if active_rows.len() > 1
                && active_rows.iter().any(|active| !*active)
                && state.service_group.is_none()
            {
                self.materialize_workflow_value(values, &carry.next)?;
                self.materialize_workflow_value(values, &carry.body_output)?;
            }
            if state.service_group.is_some() && !values.contains_key(&carry.body_output) {
                final_state_refs.insert(carry.cell.clone(), carry.next.clone());
                continue;
            }
            {
                let current = values.get(&carry.next).with_context(|| {
                    format!("workflow loop value '{}' is unavailable", carry.next)
                })?;
                let next = values.get(&carry.body_output).with_context(|| {
                    format!("workflow loop body did not produce '{}'", carry.body_output)
                })?;
                validate_state_recurrence(&carry.cell, current, next, state, values)?;
            }
            let next_value =
                if active_rows.iter().all(|active| *active) || state.service_group.is_some() {
                    // Service-managed state preserves inactive rows
                    // through its logical lengths. Retain the device
                    // value directly instead of materializing it on the
                    // host merely to clone the same tensor.
                    share_workflow_value(values, &carry.body_output)?
                } else {
                    let current = values.get(&carry.next).with_context(|| {
                        format!("workflow loop value '{}' is unavailable", carry.next)
                    })?;
                    let next = values.get(&carry.body_output).with_context(|| {
                        format!("workflow loop body did not produce '{}'", carry.body_output)
                    })?;
                    merge_inactive_rows(current, next, &active_rows, state).with_context(|| {
                        format!(
                            "workflow loop carry '{}' cannot preserve inactive rows",
                            carry.cell
                        )
                    })?
                };
            values.insert(carry.next.clone(), next_value);
            final_state_refs.insert(carry.cell.clone(), carry.next.clone());
        }
        Ok(true)
    }
}

/// A declared generation loop, driven one iteration at a time.
///
/// The scheduler's prioritized drive interleaves several requests, advancing
/// each by one token before moving on. It therefore cannot call a function that
/// runs a loop to completion — but it must still run *the same loop*, or a
/// prioritized request would obey different declared semantics than a
/// standalone one.
///
/// This owns exactly the state an in-flight interpretation holds between
/// iterations and drives it through [`WorkflowRuntime::advance_workflow_loop`],
/// the same method the run-to-completion path uses. Nothing about the loop is
/// restated here: the bound, the liveness predicate, the carries and the emits
/// all come from the compiled graph.
pub(crate) struct WorkflowGenerationCursor {
    values: PipelineTensors,
    symbols: HashMap<String, i64>,
    dynamic_symbols: std::collections::HashSet<String>,
    emit_counts: HashMap<String, usize>,
    final_state_refs: HashMap<String, String>,
    component_overrides: HashMap<String, String>,
    telemetry: WorkflowRunTelemetry,
    limit: usize,
    index: usize,
}

impl WorkflowGenerationCursor {
    /// Bind a request and run the loop's setup, stopping before its first
    /// iteration.
    pub(crate) fn start(
        runtime: &WorkflowRuntime,
        request: PipelineGenerateRequest,
        hosted: &[&str],
        host: &mut Option<&mut dyn WorkflowNodeHost>,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            runtime.adapter_service.is_none(),
            "the per-token drive cannot advance a package that selects adapters per request; \
             adapter selection is prepared once for a whole pass"
        );
        anyhow::ensure!(
            runtime.execution_islands.is_empty(),
            "the per-token drive cannot advance a package with fused execution islands, whose \
             bindings are established for a whole pass"
        );
        anyhow::ensure!(
            !runtime
                .workflow
                .state
                .values()
                .any(|cell| cell.scope == onnx_genai_metadata::WorkflowStateScope::Session),
            "the per-token drive cannot advance a package with session-scoped workflow state, \
             which is published when a pass completes"
        );
        let loop_node = sole_generation_loop(&runtime.compiled_workflow.graph)?;
        anyhow::ensure!(
            matches!(
                loop_plan(loop_node)?.setup,
                WorkflowNode::Sequence { nodes } if nodes.is_empty()
            ),
            "the per-token drive cannot advance a loop with setup steps: setup runs once per \
             pass, and a drive that hands out one iteration at a time has no pass to run it in"
        );
        let mut plan = WorkflowExecutionPlan::new_hosted(runtime, request, hosted)?;
        let mut telemetry = WorkflowRunTelemetry {
            started: Some(std::time::Instant::now()),
            max_iterations_only: plan.max_iterations_only,
            ..WorkflowRunTelemetry::default()
        };
        let mut values = std::mem::take(&mut plan.values);
        let mut symbols = plan.initial_symbols.clone();
        let dynamic_symbols = plan.dynamic_symbols.clone();
        let component_overrides = plan.component_overrides.clone();
        let mut emit_counts = HashMap::new();
        let mut final_state_refs = HashMap::new();
        let plan_parts = loop_plan(loop_node)?;
        let limit = runtime.begin_workflow_loop(
            &plan_parts,
            &runtime.workflow,
            &mut values,
            &mut symbols,
            &dynamic_symbols,
            &mut emit_counts,
            &mut final_state_refs,
            &component_overrides,
            &mut telemetry,
            host,
        )?;
        Ok(Self {
            values,
            symbols,
            dynamic_symbols,
            emit_counts,
            final_state_refs,
            component_overrides,
            telemetry,
            limit,
            index: 0,
        })
    }

    /// Run one iteration. `false` means the loop ended, by predicate or bound.
    pub(crate) fn advance(
        &mut self,
        runtime: &WorkflowRuntime,
        host: &mut Option<&mut dyn WorkflowNodeHost>,
    ) -> anyhow::Result<bool> {
        if self.index >= self.limit {
            return Ok(false);
        }
        let loop_node = sole_generation_loop(&runtime.compiled_workflow.graph)?;
        let plan_parts = loop_plan(loop_node)?;
        let ran = runtime.advance_workflow_loop(
            &plan_parts,
            &runtime.workflow,
            self.index,
            &mut self.values,
            &mut self.symbols,
            &self.dynamic_symbols,
            &mut self.emit_counts,
            &mut self.final_state_refs,
            &self.component_overrides,
            &mut self.telemetry,
            host,
        )?;
        if !ran {
            self.index = self.limit;
            return Ok(false);
        }
        self.index += 1;
        Ok(true)
    }

    /// Release the loop's induction value and report the emitted token stream.
    pub(crate) fn finish(
        mut self,
        runtime: &WorkflowRuntime,
    ) -> anyhow::Result<(PipelineTensors, bool)> {
        let loop_node = sole_generation_loop(&runtime.compiled_workflow.graph)?;
        end_workflow_loop(&loop_plan(loop_node)?, &mut self.values);
        runtime.publish_workflow_telemetry(self.telemetry);
        let ended_by_predicate = runtime.last_generation_ended_by_predicate();
        Ok((self.values, ended_by_predicate))
    }

    /// The token stream this interpretation has emitted so far.
    ///
    /// Read through the package's declared outputs, exactly as a completed pass
    /// is: an emit into a row-wise output lands on its rows, and reading the
    /// bare name would find nothing and call that "emitted nothing".
    pub(crate) fn emitted_tokens(&self, runtime: &WorkflowRuntime) -> anyhow::Result<Vec<i64>> {
        let Some(output) = runtime
            .workflow
            .outputs
            .iter()
            .find(|(_, output)| output.role == onnx_genai_metadata::WorkflowOutputRole::Tokens)
            .map(|(name, _)| name.clone())
        else {
            return Ok(Vec::new());
        };
        let values = clone_pipeline_tensors(&self.values)?;
        let outputs = runtime.package_outputs(values, self.telemetry.row_outputs.clone())?;
        let rows = outputs.rows(&output);
        let Some(value) = outputs
            .aggregate(&output)
            .or_else(|| rows.first().map(|(_, value)| *value))
        else {
            return Ok(Vec::new());
        };
        value.to_vec_i64().map_err(Into::into)
    }

    /// Tokens the loop emitted, per request-aligned row.
    ///
    /// A row-major body's emit writes one accumulator per row and no aggregate,
    /// so reading the aggregate — or falling back to row 0 — would report one
    /// row's stream as the whole batch's. This says what each row published, so
    /// a caller can hold the declared emit against what its executor committed
    /// *per row*, which is the only comparison that catches a row whose tokens
    /// stopped reaching the stream.
    pub(crate) fn emitted_token_rows(
        &self,
        runtime: &WorkflowRuntime,
    ) -> anyhow::Result<Vec<Vec<i64>>> {
        let Some(output) = runtime
            .workflow
            .outputs
            .iter()
            .find(|(_, output)| output.role == onnx_genai_metadata::WorkflowOutputRole::Tokens)
            .map(|(name, _)| name.clone())
        else {
            return Ok(Vec::new());
        };
        let values = clone_pipeline_tensors(&self.values)?;
        let outputs = runtime.package_outputs(values, self.telemetry.row_outputs.clone())?;
        let rows = outputs.rows(&output);
        if rows.is_empty() {
            return Ok(outputs
                .aggregate(&output)
                .map(|value| value.to_vec_i64())
                .transpose()?
                .into_iter()
                .collect());
        }
        let mut emitted = vec![Vec::new(); rows.iter().map(|(row, _)| row + 1).max().unwrap_or(0)];
        for (row, value) in rows {
            emitted[row] = value.to_vec_i64()?;
        }
        Ok(emitted)
    }
}

/// The one generation loop a compiled graph declares.
///
/// A package the per-token drive can advance is a generation loop and nothing
/// else: any node beside it would run once per pass, and a drive that advances
/// one iteration at a time has no pass to run it in. Saying so is what keeps
/// the prioritized path from quietly skipping a declared step.
fn sole_generation_loop(graph: &WorkflowNode) -> anyhow::Result<&WorkflowNode> {
    match graph {
        WorkflowNode::Loop { .. } => Ok(graph),
        WorkflowNode::Sequence { nodes } => {
            let [only] = nodes.as_slice() else {
                anyhow::bail!(
                    "the per-token drive requires a workflow that is one generation loop; this \
                     one declares {} top-level steps, which a per-iteration drive has no pass to \
                     run",
                    nodes.len()
                )
            };
            sole_generation_loop(only)
        }
        other => anyhow::bail!(
            "the per-token drive requires a workflow that is one generation loop; found \
             {other:?}"
        ),
    }
}

fn loop_plan(node: &WorkflowNode) -> anyhow::Result<WorkflowLoopPlan<'_>> {
    let WorkflowNode::Loop {
        setup,
        body,
        continue_when,
        max_iterations,
        termination,
        iteration,
        carried,
        ..
    } = node
    else {
        anyhow::bail!("expected a loop node");
    };
    Ok(WorkflowLoopPlan {
        setup,
        body,
        continue_when,
        max_iterations,
        termination,
        iteration,
        carried,
    })
}

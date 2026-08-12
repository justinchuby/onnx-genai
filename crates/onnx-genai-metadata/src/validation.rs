//! Validate metadata against runtime capabilities.

use std::collections::{BTreeMap, BTreeSet};

use crate::schema::{
    ControlFlow, InferenceMetadata, PipelineSpec, PipelineStrategy, PipelineStrategyKind,
    ProgramOperation, StateScope, Termination, WorkflowNode, WorkflowSpec,
};

/// Capabilities this runtime supports.
pub struct RuntimeCapabilities {
    pub supported: Vec<String>,
}

impl Default for RuntimeCapabilities {
    fn default() -> Self {
        Self {
            supported: vec![
                "kv_cache".to_string(),
                "grouped_query_attention".to_string(),
                "multi_head_attention".to_string(),
                "prefix_cache".to_string(),
                "continuous_batching".to_string(),
                "control_flow_loop".to_string(),
            ],
        }
    }
}

/// Validate the metadata document and required runtime capabilities.
pub fn validate(
    metadata: &InferenceMetadata,
    runtime: &RuntimeCapabilities,
) -> Result<(), Vec<String>> {
    let mut errors = validate_metadata(metadata).err().unwrap_or_default();
    let required = metadata
        .required_capabilities
        .iter()
        .cloned()
        .chain(derived_capabilities(metadata));
    errors.extend(required.filter(|capability| !runtime.supported.contains(capability)));

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Capabilities implied by concrete metadata features.
pub fn derived_capabilities(metadata: &InferenceMetadata) -> BTreeSet<String> {
    let mut capabilities = BTreeSet::new();
    let Some(pipeline) = &metadata.pipeline else {
        return capabilities;
    };
    if let Some(workflow) = &pipeline.workflow {
        capabilities.extend(workflow.manifest.capabilities.iter().cloned());
        capabilities.insert("workflow_ssa".to_string());
        capabilities.insert("linear_effects".to_string());
        if workflow.serving.is_some() {
            capabilities.insert("serving_service_contract".to_string());
        }
        if workflow
            .state
            .values()
            .any(|state| state.scope == crate::schema::WorkflowStateScope::Session)
        {
            capabilities.insert("session_state_lease".to_string());
        }
        collect_workflow_capabilities(&workflow.graph, &mut capabilities);
        return capabilities;
    }
    if let Some(control) = &pipeline.control {
        collect_control_capabilities(control, &mut capabilities);
    }

    if pipeline.reducers.values().any(|reducer| {
        !matches!(
            reducer.kind,
            crate::schema::ReducerKind::First | crate::schema::ReducerKind::Last
        )
    }) {
        capabilities.insert("tensor_reducers".to_string());
    }
    if pipeline
        .states
        .values()
        .any(|state| state.scope == StateScope::Session)
    {
        capabilities.insert("persistent_session_state".to_string());
    }
    for program in pipeline.programs.values() {
        for operation in &program.operations {
            match operation {
                ProgramOperation::Sample { .. } => {
                    capabilities.insert("sampling_program".to_string());
                }
                ProgramOperation::SolverStep { .. } => {
                    capabilities.insert("solver_program".to_string());
                }
                ProgramOperation::Copy { .. } | ProgramOperation::Cast { .. } => {}
            }
        }
    }
    if pipeline
        .batching
        .as_ref()
        .is_some_and(|batching| batching.continuous)
    {
        capabilities.insert("continuous_batching".to_string());
    }
    if pipeline.postprocessing.is_some() {
        capabilities.insert("postprocessing_program".to_string());
    }
    capabilities
}

fn collect_workflow_capabilities(node: &WorkflowNode, capabilities: &mut BTreeSet<String>) {
    match node {
        WorkflowNode::Sequence { nodes } => {
            for node in nodes {
                collect_workflow_capabilities(node, capabilities);
            }
        }
        WorkflowNode::Invoke { effects, .. } => {
            if !effects.is_empty() {
                capabilities.insert("linear_effects".to_string());
            }
        }
        WorkflowNode::Loop { setup, body, .. } => {
            capabilities.insert("nested_control_flow".to_string());
            collect_workflow_capabilities(setup, capabilities);
            collect_workflow_capabilities(body, capabilities);
        }
        WorkflowNode::Branch { cases, default, .. } => {
            capabilities.insert("nested_control_flow".to_string());
            for case in cases.values() {
                collect_workflow_capabilities(case, capabilities);
            }
            if let Some(default) = default {
                collect_workflow_capabilities(default, capabilities);
            }
        }
        WorkflowNode::Emit { mode, .. } => {
            capabilities.insert("typed_emit".to_string());
            if matches!(mode, crate::schema::WorkflowEmitMode::Event) {
                capabilities.insert("streaming_emit".to_string());
            }
        }
        WorkflowNode::Transfer { .. } => {
            capabilities.insert("explicit_transfer".to_string());
        }
    }
}

fn collect_control_capabilities(control: &ControlFlow, capabilities: &mut BTreeSet<String>) {
    match control {
        ControlFlow::Sequence { steps } => {
            for step in steps {
                collect_control_capabilities(step, capabilities);
            }
        }
        ControlFlow::Invoke { .. } => {}
        ControlFlow::Loop {
            body, termination, ..
        } => {
            capabilities.insert("control_flow_loop".to_string());
            if matches!(termination, Termination::Predicate { .. }) {
                capabilities.insert("predicate_termination".to_string());
            }
            collect_control_capabilities(body, capabilities);
        }
        ControlFlow::Branch { cases, default, .. } => {
            capabilities.insert("control_flow_branch".to_string());
            for case in cases.values() {
                collect_control_capabilities(case, capabilities);
            }
            if let Some(default) = default {
                collect_control_capabilities(default, capabilities);
            }
        }
    }
}

/// Validate document-level invariants independent of runtime capabilities.
pub fn validate_metadata(metadata: &InferenceMetadata) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    if let Err(error) = validate_composite_io(metadata) {
        errors.push(error);
    }

    if let Some(pipeline) = &metadata.pipeline
        && let Err(error) = validate_pipeline_spec(pipeline)
    {
        errors.extend(error.errors);
    }
    validate_preprocessing_workflow(metadata, &mut errors);

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn validate_preprocessing_workflow(metadata: &InferenceMetadata, errors: &mut Vec<String>) {
    const IMAGE_ABI: &str = "onnx-genai.image-preprocess";
    const IMAGE_ABI_VERSION: &str = "1";

    let Some(workflow) = metadata
        .pipeline
        .as_ref()
        .and_then(|pipeline| pipeline.workflow.as_ref())
    else {
        return;
    };
    let image = metadata
        .preprocessing
        .as_ref()
        .and_then(|spec| spec.image.as_ref());
    let adapters = workflow
        .components
        .iter()
        .filter(|(_, component)| {
            matches!(
                &component.implementation,
                crate::schema::ComponentImplementation::Adapter { abi, version, .. }
                    if abi == IMAGE_ABI && version == IMAGE_ABI_VERSION
            )
        })
        .collect::<Vec<_>>();
    if image.is_none() && !adapters.is_empty() {
        errors.push(format!(
            "workflow adapter components using {IMAGE_ABI}@{IMAGE_ABI_VERSION} require \
                 preprocessing.image metadata"
        ));
        return;
    }
    let Some(image) = image else {
        return;
    };
    if adapters.len() != 1 {
        errors.push(format!(
            "preprocessing.image requires exactly one workflow adapter component using \
                 {IMAGE_ABI}@{IMAGE_ABI_VERSION}, found {}",
            adapters.len()
        ));
        return;
    }
    let (adapter_name, adapter) = adapters[0];
    match adapter.ports.inputs.get("encoded") {
        Some(contract) if contract.dtype == "uint8" && contract.rank == 1 => {}
        Some(contract) => errors.push(format!(
            "workflow image preprocessing adapter '{adapter_name}' input 'encoded' must be uint8 \
                 rank 1, got {} rank {}",
            contract.dtype, contract.rank
        )),
        None => errors.push(format!(
            "workflow image preprocessing adapter '{adapter_name}' must declare input 'encoded'"
        )),
    }

    fn collect_invocations<'a>(
        node: &'a WorkflowNode,
        component: &str,
        outputs: &mut Vec<&'a BTreeMap<String, String>>,
    ) {
        match node {
            WorkflowNode::Sequence { nodes } => {
                for node in nodes {
                    collect_invocations(node, component, outputs);
                }
            }
            WorkflowNode::Invoke {
                component: invoked,
                outputs: invoked_outputs,
                ..
            } if invoked == component => outputs.push(invoked_outputs),
            WorkflowNode::Loop { setup, body, .. } => {
                collect_invocations(setup, component, outputs);
                collect_invocations(body, component, outputs);
            }
            WorkflowNode::Branch { cases, default, .. } => {
                for case in cases.values() {
                    collect_invocations(case, component, outputs);
                }
                if let Some(default) = default {
                    collect_invocations(default, component, outputs);
                }
            }
            WorkflowNode::Invoke { .. }
            | WorkflowNode::Emit { .. }
            | WorkflowNode::Transfer { .. } => {}
        }
    }
    let mut invocations = Vec::new();
    collect_invocations(&workflow.graph, adapter_name, &mut invocations);
    if invocations.len() != 1 {
        errors.push(format!(
            "workflow image preprocessing adapter '{adapter_name}' must be invoked exactly once, \
                 found {} invocations",
            invocations.len()
        ));
        return;
    }
    let invocation_outputs = invocations[0];
    for output in &image.outputs {
        if output.optional.unwrap_or(false) {
            errors.push(format!(
                "preprocessing.image output '{}' cannot be optional in a workflow; every declared \
                 adapter SSA output must be materialized",
                output.name
            ));
        }
        let Some(contract) = &output.contract else {
            errors.push(format!(
                "preprocessing.image output '{}' must declare a TensorContract for workflow use",
                output.name
            ));
            continue;
        };
        if contract.dtype != output.dtype {
            errors.push(format!(
                "preprocessing.image output '{}' dtype '{}' disagrees with its TensorContract '{}'",
                output.name, output.dtype, contract.dtype
            ));
        }
        let port = invocation_outputs
            .iter()
            .find_map(|(port, value)| (value == &output.name).then_some(port));
        let Some(port) = port else {
            errors.push(format!(
                "preprocessing.image output '{}' must be a declared SSA output of adapter \
                     invocation '{adapter_name}'",
                output.name
            ));
            continue;
        };
        match adapter.ports.outputs.get(port) {
            Some(port_contract) => require_compatible_tensor_contracts(
                contract,
                port_contract,
                &format!("preprocessing.image output '{}'", output.name),
                errors,
            ),
            None => errors.push(format!(
                "workflow image preprocessing adapter '{adapter_name}' has no output port '{port}'"
            )),
        }
    }
}

fn require_compatible_tensor_contracts(
    source: &crate::schema::TensorContract,
    target: &crate::schema::TensorContract,
    path: &str,
    errors: &mut Vec<String>,
) {
    fn normalize(dtype: &str) -> &str {
        match dtype {
            "fp32" => "float32",
            "fp16" => "float16",
            "bf16" => "bfloat16",
            other => other,
        }
    }
    if normalize(&source.dtype) != normalize(&target.dtype)
        || source.rank != target.rank
        || source.shape != target.shape
    {
        errors.push(format!(
            "{path} has a contract incompatible with its adapter output port"
        ));
    }
}

pub(crate) fn validate_composite_io(metadata: &InferenceMetadata) -> Result<(), String> {
    if metadata.pipeline.is_some()
        && metadata
            .model
            .as_ref()
            .and_then(|model| model.io.as_ref())
            .is_some()
    {
        Err(
            "model.io is only valid for bare single-model metadata; when pipeline is present, \
             declare decoder I/O at pipeline.models.<component>.io"
                .to_string(),
        )
    } else {
        Ok(())
    }
}

/// All structural problems found in a pipeline specification.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid pipeline spec: {errors:?}")]
pub struct PipelineValidationError {
    pub errors: Vec<String>,
}

/// Validate the pipeline DAG and component references.
pub fn validate_pipeline_spec(spec: &PipelineSpec) -> Result<(), PipelineValidationError> {
    let mut errors = Vec::new();

    if let Some(workflow) = &spec.workflow {
        if !spec.models.is_empty()
            || !spec.dataflow.is_empty()
            || spec.control.is_some()
            || !spec.phases.is_empty()
        {
            errors.push(
                "pipeline.workflow is exclusive; legacy models/dataflow/control/phases must be removed"
                    .to_string(),
            );
        }
        validate_workflow(workflow, &mut errors);
        return if errors.is_empty() {
            Ok(())
        } else {
            Err(PipelineValidationError { errors })
        };
    }

    if spec.models.is_empty() {
        errors.push("pipeline.models must contain at least one component".to_string());
    }

    fn validate_workflow(workflow: &WorkflowSpec, errors: &mut Vec<String>) {
        if workflow.manifest.ir_version != "1.0" {
            errors.push(format!(
                "unsupported pipeline.workflow.manifest.ir_version '{}'; this runtime supports 1.0",
                workflow.manifest.ir_version
            ));
        }
        if workflow.manifest.onnx_opsets.is_empty() {
            errors.push("pipeline.workflow.manifest.onnx_opsets must not be empty".to_string());
        }
        for (domain, version) in &workflow.manifest.onnx_opsets {
            if domain.trim().is_empty() || *version == 0 {
                errors.push(format!(
                    "pipeline.workflow.manifest.onnx_opsets contains invalid {domain:?}@{version}"
                ));
            }
        }
        fn validate_policy_component(
            name: &str,
            component: &crate::schema::WorkflowComponent,
            errors: &mut Vec<String>,
        ) {
            use crate::schema::{PolicyComponentContract as Policy, SamplingPolicyMode};

            let Some(policy) = &component.policy else {
                return;
            };
            for (port, contract) in component
                .ports
                .inputs
                .iter()
                .chain(component.ports.outputs.iter())
            {
                require_declared_shape(name, port, contract, errors);
            }
            let input = |port: &str| component.ports.inputs.get(port);
            let output = |port: &str| component.ports.outputs.get(port);
            let require_input = |port: &str, errors: &mut Vec<String>| {
                input(port).or_else(|| {
                    errors.push(format!(
                        "workflow policy component '{name}' references missing input port '{port}'"
                    ));
                    None
                })
            };
            let require_output = |port: &str, errors: &mut Vec<String>| {
                output(port).or_else(|| {
                    errors.push(format!(
                        "workflow policy component '{name}' references missing output port '{port}'"
                    ));
                    None
                })
            };
            let require_effect = |effect: &str, errors: &mut Vec<String>| {
                if !component.effects.iter().any(|declared| declared == effect) {
                    errors.push(format!(
                        "workflow policy component '{name}' references undeclared effect '{effect}'"
                    ));
                }
            };
            let validate_rng = |rng: &crate::schema::RngPortContract, errors: &mut Vec<String>| {
                for port in [&rng.seed, &rng.offset] {
                    if let Some(contract) = require_input(port, errors) {
                        require_dtype_rank(name, port, contract, "int64", 1, errors);
                    }
                }
                if let Some(contract) = require_output(&rng.next_offset, errors) {
                    require_dtype_rank(name, &rng.next_offset, contract, "int64", 1, errors);
                }
                require_same_contract(
                    name,
                    &rng.seed,
                    input(&rng.seed),
                    &rng.offset,
                    input(&rng.offset),
                    errors,
                );
                require_same_contract(
                    name,
                    &rng.seed,
                    input(&rng.seed),
                    &rng.next_offset,
                    output(&rng.next_offset),
                    errors,
                );
            };

            match policy {
                Policy::TokenSampler {
                    mode,
                    logits,
                    token,
                    temperature,
                    top_k,
                    top_p,
                    rng,
                    effect,
                } => {
                    if let Some(contract) = require_input(logits, errors) {
                        require_floating(name, logits, contract, errors);
                        require_rank(name, logits, contract, 2, errors);
                    }
                    if let Some(contract) = require_output(token, errors) {
                        require_integer(name, token, contract, errors);
                        require_rank(name, token, contract, 1, errors);
                    }
                    require_axis_equal(
                        name,
                        logits,
                        input(logits),
                        0,
                        token,
                        output(token),
                        0,
                        errors,
                    );
                    for port in [temperature.as_ref(), top_p.as_ref()].into_iter().flatten() {
                        if let Some(contract) = require_input(port, errors) {
                            require_floating(name, port, contract, errors);
                            require_rank(name, port, contract, 1, errors);
                            require_axis_equal(
                                name,
                                logits,
                                input(logits),
                                0,
                                port,
                                Some(contract),
                                0,
                                errors,
                            );
                        }
                    }
                    if let Some(port) = top_k
                        && let Some(contract) = require_input(port, errors)
                    {
                        require_integer(name, port, contract, errors);
                        require_rank(name, port, contract, 1, errors);
                        require_axis_equal(
                            name,
                            logits,
                            input(logits),
                            0,
                            port,
                            Some(contract),
                            0,
                            errors,
                        );
                    }
                    match (mode, rng) {
                    (SamplingPolicyMode::Greedy, Some(_)) => errors.push(format!(
                        "workflow policy component '{name}' greedy sampler must not declare RNG ports"
                    )),
                    (SamplingPolicyMode::SeededStochastic, None) => errors.push(format!(
                        "workflow policy component '{name}' seeded stochastic sampler requires RNG ports"
                    )),
                    (_, Some(rng)) => validate_rng(rng, errors),
                    (_, None) => {}
                }
                    if let Some(rng) = rng {
                        require_axis_equal(
                            name,
                            logits,
                            input(logits),
                            0,
                            &rng.seed,
                            input(&rng.seed),
                            0,
                            errors,
                        );
                    }
                    require_effect(effect, errors);
                }
                Policy::TerminationPredicate {
                    tokens,
                    eos_ids,
                    iteration,
                    max_iterations,
                    done,
                    effect,
                } => {
                    for port in [tokens, eos_ids, iteration, max_iterations] {
                        if let Some(contract) = require_input(port, errors) {
                            require_integer(name, port, contract, errors);
                            require_rank(name, port, contract, 1, errors);
                        }
                    }
                    if let Some(contract) = require_output(done, errors) {
                        require_bool(name, done, contract, errors);
                        require_rank(name, done, contract, 1, errors);
                    }
                    for port in [iteration, max_iterations, done] {
                        require_axis_equal(
                            name,
                            tokens,
                            input(tokens),
                            0,
                            port,
                            input(port).or_else(|| output(port)),
                            0,
                            errors,
                        );
                    }
                    require_effect(effect, errors);
                }
                Policy::SolverStep {
                    state,
                    estimate,
                    step,
                    schedule,
                    next_state,
                    effect,
                } => {
                    let state_contract = require_input(state, errors);
                    let estimate_contract = require_input(estimate, errors);
                    let next_contract = require_output(next_state, errors);
                    for (port, contract) in [(state, state_contract), (estimate, estimate_contract)]
                    {
                        if let Some(contract) = contract {
                            require_floating(name, port, contract, errors);
                        }
                    }
                    require_same_contract(
                        name,
                        state,
                        state_contract,
                        estimate,
                        estimate_contract,
                        errors,
                    );
                    if let Some(contract) = require_input(step, errors) {
                        require_integer(name, step, contract, errors);
                        require_rank(name, step, contract, 1, errors);
                        require_axis_equal(
                            name,
                            state,
                            state_contract,
                            0,
                            step,
                            Some(contract),
                            0,
                            errors,
                        );
                    }
                    if let Some(contract) = require_input(schedule, errors) {
                        require_floating(name, schedule, contract, errors);
                        if contract.rank != 1 {
                            errors.push(format!(
                            "workflow policy component '{name}' schedule port '{schedule}' must have rank 1"
                        ));
                        }
                    }
                    require_same_contract(
                        name,
                        state,
                        state_contract,
                        next_state,
                        next_contract,
                        errors,
                    );
                    require_effect(effect, errors);
                }
                Policy::MaskedUpdate {
                    state,
                    proposal,
                    mask,
                    step,
                    next_state,
                    next_mask,
                    rng,
                    effect,
                } => {
                    let state_contract = require_input(state, errors);
                    let proposal_contract = require_input(proposal, errors);
                    let next_contract = require_output(next_state, errors);
                    for (port, contract) in [(state, state_contract), (proposal, proposal_contract)]
                    {
                        if let Some(contract) = contract {
                            require_integer(name, port, contract, errors);
                            require_rank(name, port, contract, 2, errors);
                        }
                    }
                    require_same_contract(
                        name,
                        state,
                        state_contract,
                        proposal,
                        proposal_contract,
                        errors,
                    );
                    require_same_contract(
                        name,
                        state,
                        state_contract,
                        next_state,
                        next_contract,
                        errors,
                    );
                    if let Some(contract) = require_input(mask, errors) {
                        require_bool(name, mask, contract, errors);
                        require_rank(name, mask, contract, 2, errors);
                        require_same_shape(
                            name,
                            state,
                            state_contract,
                            mask,
                            Some(contract),
                            errors,
                        );
                    }
                    if let Some(contract) = require_output(next_mask, errors) {
                        require_bool(name, next_mask, contract, errors);
                        require_rank(name, next_mask, contract, 2, errors);
                        require_same_shape(
                            name,
                            state,
                            state_contract,
                            next_mask,
                            Some(contract),
                            errors,
                        );
                    }
                    if let Some(contract) = require_input(step, errors) {
                        require_integer(name, step, contract, errors);
                        require_rank(name, step, contract, 1, errors);
                        require_axis_equal(
                            name,
                            state,
                            state_contract,
                            0,
                            step,
                            Some(contract),
                            0,
                            errors,
                        );
                    }
                    if let Some(rng) = rng {
                        validate_rng(rng, errors);
                        require_axis_equal(
                            name,
                            state,
                            state_contract,
                            0,
                            &rng.seed,
                            input(&rng.seed),
                            0,
                            errors,
                        );
                    }
                    require_effect(effect, errors);
                }
                Policy::SpeculativeVerifier {
                    target_scores,
                    proposed_tokens,
                    proposal_scores,
                    accepted_tokens,
                    accepted_len,
                    done,
                    rng,
                    effect,
                } => {
                    if let Some(contract) = require_input(target_scores, errors) {
                        require_floating(name, target_scores, contract, errors);
                        require_rank(name, target_scores, contract, 3, errors);
                    }
                    if let Some(port) = proposal_scores
                        && let Some(contract) = require_input(port, errors)
                    {
                        require_floating(name, port, contract, errors);
                        require_rank(name, port, contract, 3, errors);
                        require_same_contract(
                            name,
                            target_scores,
                            input(target_scores),
                            port,
                            Some(contract),
                            errors,
                        );
                    }
                    for (port, contract) in [
                        (proposed_tokens, require_input(proposed_tokens, errors)),
                        (accepted_tokens, require_output(accepted_tokens, errors)),
                        (accepted_len, require_output(accepted_len, errors)),
                    ] {
                        if let Some(contract) = contract {
                            require_integer(name, port, contract, errors);
                            require_rank(
                                name,
                                port,
                                contract,
                                if port == accepted_len { 1 } else { 2 },
                                errors,
                            );
                        }
                    }
                    if let Some(contract) = require_output(done, errors) {
                        require_bool(name, done, contract, errors);
                        require_rank(name, done, contract, 1, errors);
                    }
                    require_same_contract(
                        name,
                        proposed_tokens,
                        input(proposed_tokens),
                        accepted_tokens,
                        output(accepted_tokens),
                        errors,
                    );
                    for port in [accepted_len, done] {
                        require_axis_equal(
                            name,
                            proposed_tokens,
                            input(proposed_tokens),
                            0,
                            port,
                            output(port),
                            0,
                            errors,
                        );
                    }
                    for (left_axis, right_axis) in [(0, 0), (1, 1)] {
                        require_axis_equal(
                            name,
                            target_scores,
                            input(target_scores),
                            left_axis,
                            proposed_tokens,
                            input(proposed_tokens),
                            right_axis,
                            errors,
                        );
                    }
                    if let Some(rng) = rng {
                        validate_rng(rng, errors);
                        require_axis_equal(
                            name,
                            proposed_tokens,
                            input(proposed_tokens),
                            0,
                            &rng.seed,
                            input(&rng.seed),
                            0,
                            errors,
                        );
                    }
                    require_effect(effect, errors);
                }
                Policy::StateUpdate {
                    current,
                    update,
                    next,
                    effect,
                } => {
                    let current_contract = require_input(current, errors);
                    let _ = require_input(update, errors);
                    let next_contract = require_output(next, errors);
                    require_same_contract(
                        name,
                        current,
                        current_contract,
                        next,
                        next_contract,
                        errors,
                    );
                    require_effect(effect, errors);
                }
            }
        }

        fn require_dtype_rank(
            component: &str,
            port: &str,
            contract: &crate::schema::TensorContract,
            dtype: &str,
            rank: usize,
            errors: &mut Vec<String>,
        ) {
            if contract.dtype != dtype || contract.rank != rank {
                errors.push(format!(
                "workflow policy component '{component}' port '{port}' must be {dtype} rank {rank}, \
                 got {} rank {}",
                contract.dtype, contract.rank
            ));
            }
        }

        fn require_floating(
            component: &str,
            port: &str,
            contract: &crate::schema::TensorContract,
            errors: &mut Vec<String>,
        ) {
            if !matches!(
                contract.dtype.as_str(),
                "float16" | "fp16" | "float32" | "fp32" | "float64" | "bfloat16" | "bf16"
            ) {
                errors.push(format!(
                    "workflow policy component '{component}' port '{port}' must be floating point"
                ));
            }
        }

        fn require_rank(
            component: &str,
            port: &str,
            contract: &crate::schema::TensorContract,
            rank: usize,
            errors: &mut Vec<String>,
        ) {
            if contract.rank != rank {
                errors.push(format!(
                    "workflow policy component '{component}' port '{port}' must have rank {rank}, \
                         got {}",
                    contract.rank
                ));
            }
        }

        fn require_declared_shape(
            component: &str,
            port: &str,
            contract: &crate::schema::TensorContract,
            errors: &mut Vec<String>,
        ) {
            match &contract.shape {
                Some(shape) if shape.len() == contract.rank => {}
                Some(shape) => errors.push(format!(
                    "workflow policy component '{component}' port '{port}' declares rank {} \
                     but has {} shape dimensions",
                    contract.rank,
                    shape.len()
                )),
                None => errors.push(format!(
                    "workflow policy component '{component}' port '{port}' must declare its shape"
                )),
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn require_axis_equal(
            component: &str,
            left_port: &str,
            left: Option<&crate::schema::TensorContract>,
            left_axis: usize,
            right_port: &str,
            right: Option<&crate::schema::TensorContract>,
            right_axis: usize,
            errors: &mut Vec<String>,
        ) {
            let left_dimension = left
                .and_then(|contract| contract.shape.as_ref())
                .and_then(|shape| shape.get(left_axis));
            let right_dimension = right
                .and_then(|contract| contract.shape.as_ref())
                .and_then(|shape| shape.get(right_axis));
            if let (Some(left_dimension), Some(right_dimension)) = (left_dimension, right_dimension)
                && left_dimension != right_dimension
            {
                errors.push(format!(
                    "workflow policy component '{component}' ports '{left_port}' axis \
                     {left_axis} and '{right_port}' axis {right_axis} must use the same dimension"
                ));
            }
        }

        fn require_integer(
            component: &str,
            port: &str,
            contract: &crate::schema::TensorContract,
            errors: &mut Vec<String>,
        ) {
            if !matches!(
                contract.dtype.as_str(),
                "int8" | "int16" | "int32" | "int64" | "uint8" | "uint16" | "uint32" | "uint64"
            ) {
                errors.push(format!(
                    "workflow policy component '{component}' port '{port}' must be integer"
                ));
            }
        }

        fn require_bool(
            component: &str,
            port: &str,
            contract: &crate::schema::TensorContract,
            errors: &mut Vec<String>,
        ) {
            if contract.dtype != "bool" {
                errors.push(format!(
                    "workflow policy component '{component}' port '{port}' must be bool"
                ));
            }
        }

        fn require_same_contract(
            component: &str,
            left_port: &str,
            left: Option<&crate::schema::TensorContract>,
            right_port: &str,
            right: Option<&crate::schema::TensorContract>,
            errors: &mut Vec<String>,
        ) {
            if let (Some(left), Some(right)) = (left, right)
                && left != right
            {
                errors.push(format!(
                "workflow policy component '{component}' ports '{left_port}' and '{right_port}' \
                 must have identical tensor contracts"
            ));
            }
        }

        fn require_same_shape(
            component: &str,
            left_port: &str,
            left: Option<&crate::schema::TensorContract>,
            right_port: &str,
            right: Option<&crate::schema::TensorContract>,
            errors: &mut Vec<String>,
        ) {
            if let (Some(left), Some(right)) = (left, right)
                && (left.rank != right.rank || left.shape != right.shape)
            {
                errors.push(format!(
                    "workflow policy component '{component}' ports '{left_port}' and \
                     '{right_port}' must have identical tensor shapes"
                ));
            }
        }

        fn validate_runtime_dtype(
            path: &str,
            contract: &crate::schema::TensorContract,
            errors: &mut Vec<String>,
        ) {
            if !matches!(
                contract.dtype.as_str(),
                "float16"
                    | "fp16"
                    | "float32"
                    | "fp32"
                    | "bfloat16"
                    | "bf16"
                    | "int8"
                    | "int16"
                    | "int32"
                    | "int64"
                    | "uint8"
                    | "uint16"
                    | "uint32"
                    | "uint64"
                    | "bool"
            ) {
                errors.push(format!(
                    "{path} uses dtype '{}', which the workflow runtime does not support",
                    contract.dtype
                ));
            }
        }

        for (name, input) in &workflow.inputs {
            validate_runtime_dtype(&format!("workflow input '{name}'"), &input.contract, errors);
        }
        for (name, output) in &workflow.outputs {
            validate_runtime_dtype(
                &format!("workflow output '{name}'"),
                &output.contract,
                errors,
            );
        }
        for (name, component) in &workflow.components {
            if name.trim().is_empty() || name.contains('.') {
                errors.push(format!("workflow component name is invalid: '{name}'"));
            }
            match &component.implementation {
                crate::schema::ComponentImplementation::Onnx { artifact } => {
                    if artifact.trim().is_empty() {
                        errors.push(format!(
                            "workflow component '{name}' has an empty ONNX artifact"
                        ));
                    }
                }
                crate::schema::ComponentImplementation::Adapter {
                    abi,
                    version,
                    artifact: _,
                    custom_ops,
                } => {
                    match workflow.manifest.adapter_abis.get(abi) {
                        Some(pinned) if pinned == version => {}
                        _ => errors.push(format!(
                            "workflow component '{name}' requires adapter ABI {abi}@{version}, \
                             but the manifest does not pin that exact version"
                        )),
                    }
                    for (domain, version) in custom_ops {
                        if workflow.manifest.custom_op_versions.get(domain) != Some(version) {
                            errors.push(format!(
                                "workflow component '{name}' requires custom-op domain \
                                 {domain}@{version}, but the manifest does not pin it"
                            ));
                        }
                    }
                }
                crate::schema::ComponentImplementation::Binding => {}
            }
            for (port, contract) in &component.ports.inputs {
                validate_runtime_dtype(
                    &format!("workflow component '{name}' input '{port}'"),
                    contract,
                    errors,
                );
            }
            for (port, contract) in &component.ports.outputs {
                validate_runtime_dtype(
                    &format!("workflow component '{name}' output '{port}'"),
                    contract,
                    errors,
                );
            }
            if component.policy.is_some()
                && !matches!(
                    component.implementation,
                    crate::schema::ComponentImplementation::Onnx { .. }
                )
            {
                errors.push(format!(
                    "workflow policy component '{name}' must use an ONNX implementation"
                ));
            }
            validate_policy_component(name, component, errors);
        }
        for (name, state) in &workflow.state {
            validate_runtime_dtype(&format!("workflow state '{name}'"), &state.contract, errors);
            if state.scope == crate::schema::WorkflowStateScope::Invocation
                && state.session.is_some()
            {
                errors.push(format!(
                    "workflow state '{name}' has session lease settings but invocation scope"
                ));
            }
            if let Some(session) = &state.session {
                if session.policy != crate::schema::SessionMutationPolicy::Exclusive {
                    errors.push(format!(
                        "workflow state '{name}' requests copy-on-write session mutation, \
                         which this runtime does not support"
                    ));
                }
                if session.ttl_seconds.is_some() {
                    errors.push(format!(
                        "workflow state '{name}' declares session TTL, which this runtime \
                         does not yet enforce"
                    ));
                }
                if session.optimistic_metadata_version {
                    errors.push(format!(
                        "workflow state '{name}' requests optimistic metadata versioning, \
                         which this runtime does not support"
                    ));
                }
            }

            if let crate::schema::ShapeRecurrence::Growing { axis, .. } = &state.recurrence
                && *axis >= state.contract.rank
            {
                errors.push(format!(
                    "workflow state '{name}' grows on axis {axis}, outside rank {}",
                    state.contract.rank
                ));
            }
        }

        let mut values = workflow.inputs.keys().cloned().collect::<BTreeSet<_>>();
        let mut value_contracts = workflow
            .inputs
            .iter()
            .map(|(name, input)| (name.clone(), input.contract.clone()))
            .collect::<BTreeMap<_, _>>();
        let mut effects = workflow.initial_effects.clone();
        let mut effect_tokens = effects.values().cloned().collect::<BTreeSet<_>>();
        validate_workflow_node(
            &workflow.graph,
            workflow,
            &mut values,
            &mut value_contracts,
            &mut effects,
            &mut effect_tokens,
            "pipeline.workflow.graph",
            errors,
        );

        let mut used = BTreeSet::from(["workflow_ssa".to_string(), "linear_effects".to_string()]);
        if workflow.serving.is_some() {
            used.insert("serving_service_contract".to_string());
        }
        if workflow
            .state
            .values()
            .any(|state| state.scope == crate::schema::WorkflowStateScope::Session)
        {
            used.insert("session_state_lease".to_string());
        }
        collect_workflow_capabilities(&workflow.graph, &mut used);
        for capability in used.difference(&workflow.manifest.capabilities) {
            errors.push(format!(
                "pipeline.workflow.manifest.capabilities is missing used capability '{capability}'"
            ));
        }
    }

    fn validate_workflow_node(
        node: &WorkflowNode,
        workflow: &WorkflowSpec,
        values: &mut BTreeSet<String>,
        value_contracts: &mut BTreeMap<String, crate::schema::TensorContract>,
        effects: &mut BTreeMap<String, String>,
        effect_tokens: &mut BTreeSet<String>,
        path: &str,
        errors: &mut Vec<String>,
    ) {
        match node {
            WorkflowNode::Sequence { nodes } => {
                if nodes.is_empty() {
                    errors.push(format!("{path}.nodes must not be empty"));
                }
                for (index, node) in nodes.iter().enumerate() {
                    validate_workflow_node(
                        node,
                        workflow,
                        values,
                        value_contracts,
                        effects,
                        effect_tokens,
                        &format!("{path}.nodes[{index}]"),
                        errors,
                    );
                }
            }
            WorkflowNode::Invoke {
                component,
                inputs,
                outputs,
                effects: transitions,
            } => {
                let Some(declaration) = workflow.components.get(component) else {
                    errors.push(format!("{path} invokes unknown component '{component}'"));
                    return;
                };
                for (port, value) in inputs {
                    if !declaration.ports.inputs.contains_key(port) {
                        errors.push(format!("{path}.inputs has unknown port '{port}'"));
                    }
                    require_workflow_value(value, values, &format!("{path}.inputs.{port}"), errors);
                    if let (Some(source), Some(target)) = (
                        value_contracts.get(value),
                        declaration.ports.inputs.get(port),
                    ) {
                        require_compatible_contracts(
                            source,
                            target,
                            &format!("{path}.inputs.{port}"),
                            errors,
                        );
                    }
                }
                for port in declaration.ports.inputs.keys() {
                    if !inputs.contains_key(port) {
                        errors.push(format!("{path}.inputs is missing port '{port}'"));
                    }
                }
                for (port, value) in outputs {
                    if !declaration.ports.outputs.contains_key(port) {
                        errors.push(format!("{path}.outputs has unknown port '{port}'"));
                    }
                    define_workflow_value(value, values, &format!("{path}.outputs.{port}"), errors);
                    if let Some(contract) = declaration.ports.outputs.get(port) {
                        value_contracts.insert(value.clone(), contract.clone());
                    }
                }
                if let Some(policy) = &declaration.policy {
                    use crate::schema::PolicyComponentContract as Policy;
                    let required_outputs: Vec<&str> = match policy {
                        Policy::TokenSampler { token, rng, .. } => {
                            let mut ports = vec![token.as_str()];
                            if let Some(rng) = rng {
                                ports.push(rng.next_offset.as_str());
                            }
                            ports
                        }
                        Policy::TerminationPredicate { done, .. } => vec![done],
                        Policy::SolverStep { next_state, .. } => vec![next_state],
                        Policy::MaskedUpdate {
                            next_state,
                            next_mask,
                            rng,
                            ..
                        } => {
                            let mut ports = vec![next_state.as_str(), next_mask.as_str()];
                            if let Some(rng) = rng {
                                ports.push(rng.next_offset.as_str());
                            }
                            ports
                        }
                        Policy::SpeculativeVerifier {
                            accepted_tokens,
                            accepted_len,
                            done,
                            rng,
                            ..
                        } => {
                            let mut ports = vec![
                                accepted_tokens.as_str(),
                                accepted_len.as_str(),
                                done.as_str(),
                            ];
                            if let Some(rng) = rng {
                                ports.push(rng.next_offset.as_str());
                            }
                            ports
                        }
                        Policy::StateUpdate { next, .. } => vec![next],
                    };
                    for port in required_outputs {
                        if !outputs.contains_key(port) {
                            errors.push(format!(
                                "{path}.outputs is missing required policy output port '{port}'"
                            ));
                        }
                    }
                }
                for effect in &declaration.effects {
                    if !transitions.contains_key(effect) {
                        errors.push(format!(
                            "{path}.effects is missing declared effect '{effect}'"
                        ));
                    }
                }
                for (effect, transition) in transitions {
                    apply_effect_transition(
                        effect,
                        transition,
                        effects,
                        effect_tokens,
                        &format!("{path}.effects.{effect}"),
                        errors,
                    );
                }
            }
            WorkflowNode::Loop {
                setup,
                body,
                condition,
                max_iterations,
                carried,
            } => {
                validate_workflow_node(
                    setup,
                    workflow,
                    values,
                    value_contracts,
                    effects,
                    effect_tokens,
                    &format!("{path}.setup"),
                    errors,
                );
                require_workflow_value(
                    max_iterations,
                    values,
                    &format!("{path}.max_iterations"),
                    errors,
                );
                let mut body_values = values.clone();
                let mut body_contracts = value_contracts.clone();
                let mut body_effects = effects.clone();
                for carry in carried {
                    let Some(state) = workflow.state.get(&carry.cell) else {
                        errors.push(format!(
                            "{path}.carried references unknown cell '{}'",
                            carry.cell
                        ));
                        continue;
                    };
                    require_workflow_value(
                        &state.initializer,
                        values,
                        &format!("{path}.carried.initializer"),
                        errors,
                    );
                    if let Some(initializer_contract) = value_contracts.get(&state.initializer) {
                        require_state_contract(
                            initializer_contract,
                            &state.contract,
                            &state.recurrence,
                            false,
                            &format!("{path}.carried.initializer"),
                            errors,
                        );
                    }
                    require_workflow_value(
                        &carry.current,
                        values,
                        &format!("{path}.carried.current"),
                        errors,
                    );
                    if let Some(current_contract) = value_contracts.get(&carry.current) {
                        require_state_contract(
                            current_contract,
                            &state.contract,
                            &state.recurrence,
                            false,
                            &format!("{path}.carried.current"),
                            errors,
                        );
                    }
                    body_values.insert(carry.body_input.clone());
                    body_contracts.insert(carry.body_input.clone(), state.contract.clone());
                    apply_effect_transition(
                        &format!("state:{}", carry.cell),
                        &carry.read_effect,
                        &mut body_effects,
                        effect_tokens,
                        &format!("{path}.carried.read_effect"),
                        errors,
                    );
                }
                validate_workflow_node(
                    body,
                    workflow,
                    &mut body_values,
                    &mut body_contracts,
                    &mut body_effects,
                    effect_tokens,
                    &format!("{path}.body"),
                    errors,
                );
                require_workflow_value(
                    condition,
                    &body_values,
                    &format!("{path}.condition"),
                    errors,
                );
                for carry in carried {
                    require_workflow_value(
                        &carry.body_output,
                        &body_values,
                        &format!("{path}.carried.body_output"),
                        errors,
                    );
                    if let (Some(next_contract), Some(state)) = (
                        body_contracts.get(&carry.body_output),
                        workflow.state.get(&carry.cell),
                    ) {
                        require_state_contract(
                            next_contract,
                            &state.contract,
                            &state.recurrence,
                            true,
                            &format!("{path}.carried.body_output"),
                            errors,
                        );
                    }
                    apply_effect_transition(
                        &format!("state:{}", carry.cell),
                        &carry.write_effect,
                        &mut body_effects,
                        effect_tokens,
                        &format!("{path}.carried.write_effect"),
                        errors,
                    );
                    define_workflow_value(
                        &carry.next,
                        values,
                        &format!("{path}.carried.next"),
                        errors,
                    );
                    if let Some(contract) = body_contracts.get(&carry.body_output) {
                        value_contracts.insert(carry.next.clone(), contract.clone());
                    }
                }
                *effects = body_effects;
            }
            WorkflowNode::Branch {
                predicate,
                cases,
                default,
                outputs,
                effects: merges,
            } => {
                require_workflow_value(predicate, values, &format!("{path}.predicate"), errors);
                if cases.is_empty() {
                    errors.push(format!("{path}.cases must not be empty"));
                }
                let mut case_scopes = BTreeMap::new();
                for (case, node) in cases {
                    let mut case_values = values.clone();
                    let mut case_contracts = value_contracts.clone();
                    let mut case_effects = effects.clone();
                    let mut case_tokens = effect_tokens.clone();
                    validate_workflow_node(
                        node,
                        workflow,
                        &mut case_values,
                        &mut case_contracts,
                        &mut case_effects,
                        &mut case_tokens,
                        &format!("{path}.cases.{case}"),
                        errors,
                    );
                    case_scopes.insert(
                        case.clone(),
                        (case_values, case_contracts, case_effects, case_tokens),
                    );
                }
                let default_scope = if let Some(default) = default {
                    let mut default_values = values.clone();
                    let mut default_contracts = value_contracts.clone();
                    let mut default_effects = effects.clone();
                    let mut default_tokens = effect_tokens.clone();
                    validate_workflow_node(
                        default,
                        workflow,
                        &mut default_values,
                        &mut default_contracts,
                        &mut default_effects,
                        &mut default_tokens,
                        &format!("{path}.default"),
                        errors,
                    );
                    Some((
                        default_values,
                        default_contracts,
                        default_effects,
                        default_tokens,
                    ))
                } else {
                    None
                };

                for (output, phi) in outputs {
                    for case in cases.keys() {
                        if !phi.cases.contains_key(case) {
                            errors.push(format!(
                                "{path}.outputs.{output} has no value for case '{case}'"
                            ));
                        }
                    }
                    for case in phi.cases.keys() {
                        if !cases.contains_key(case) {
                            errors.push(format!(
                                "{path}.outputs.{output} references unknown case '{case}'"
                            ));
                        }
                    }
                    if default.is_some() != phi.default.is_some() {
                        errors.push(format!(
                            "{path}.outputs.{output} must map the default branch exactly when one \
                             is declared"
                        ));
                    }

                    let mut contract = None;
                    for (case, source) in &phi.cases {
                        let Some((case_values, case_contracts, _, _)) = case_scopes.get(case)
                        else {
                            continue;
                        };
                        require_workflow_value(
                            source,
                            case_values,
                            &format!("{path}.outputs.{output}.cases.{case}"),
                            errors,
                        );
                        if let Some(source_contract) = case_contracts.get(source) {
                            if let Some(expected) = &contract {
                                require_compatible_contracts(
                                    source_contract,
                                    expected,
                                    &format!("{path}.outputs.{output}.cases.{case}"),
                                    errors,
                                );
                            } else {
                                contract = Some(source_contract.clone());
                            }
                        }
                    }
                    if let (Some(source), Some((default_values, default_contracts, _, _))) =
                        (&phi.default, &default_scope)
                    {
                        require_workflow_value(
                            source,
                            default_values,
                            &format!("{path}.outputs.{output}.default"),
                            errors,
                        );
                        if let Some(source_contract) = default_contracts.get(source) {
                            if let Some(expected) = &contract {
                                require_compatible_contracts(
                                    source_contract,
                                    expected,
                                    &format!("{path}.outputs.{output}.default"),
                                    errors,
                                );
                            } else {
                                contract = Some(source_contract.clone());
                            }
                        }
                    }
                    define_workflow_value(
                        output,
                        values,
                        &format!("{path}.outputs.{output}"),
                        errors,
                    );
                    if let Some(contract) = contract {
                        value_contracts.insert(output.clone(), contract);
                    }
                }

                for (effect_name, merge) in merges {
                    match effects.get(effect_name) {
                        Some(incoming) if incoming == &merge.incoming => {}
                        Some(incoming) => errors.push(format!(
                            "{path}.effects.{effect_name} consumes '{}', but the incoming token is \
                             '{incoming}'",
                            merge.incoming
                        )),
                        None => errors.push(format!(
                            "{path}.effects.{effect_name} references undeclared effect"
                        )),
                    }
                    for case in cases.keys() {
                        if !merge.cases.contains_key(case) {
                            errors.push(format!(
                                "{path}.effects.{effect_name} has no successor for case '{case}'"
                            ));
                        }
                    }
                    for case in merge.cases.keys() {
                        if !cases.contains_key(case) {
                            errors.push(format!(
                                "{path}.effects.{effect_name} references unknown case '{case}'"
                            ));
                        }
                    }
                    if default.is_some() != merge.default.is_some() {
                        errors.push(format!(
                            "{path}.effects.{effect_name} must map the default branch exactly when \
                             one is declared"
                        ));
                    }
                    for (case, successor) in &merge.cases {
                        if let Some((_, _, case_effects, _)) = case_scopes.get(case)
                            && case_effects.get(effect_name) != Some(successor)
                        {
                            errors.push(format!(
                                "{path}.effects.{effect_name}.cases.{case} declares successor \
                                 '{successor}', but the case produces {:?}",
                                case_effects.get(effect_name)
                            ));
                        }
                    }
                    if let (Some(successor), Some((_, _, default_effects, _))) =
                        (&merge.default, &default_scope)
                        && default_effects.get(effect_name) != Some(successor)
                    {
                        errors.push(format!(
                            "{path}.effects.{effect_name}.default declares successor \
                             '{successor}', but the default produces {:?}",
                            default_effects.get(effect_name)
                        ));
                    }
                    if merge.cases.values().any(|token| token == &merge.produces)
                        || merge.default.as_ref() == Some(&merge.produces)
                        || case_scopes
                            .values()
                            .any(|(_, _, _, tokens)| tokens.contains(&merge.produces))
                        || default_scope
                            .as_ref()
                            .is_some_and(|(_, _, _, tokens)| tokens.contains(&merge.produces))
                    {
                        errors.push(format!(
                            "{path}.effects.{effect_name} joined successor '{}' must be distinct \
                             from every case-local successor",
                            merge.produces
                        ));
                    }
                    if !effect_tokens.insert(merge.produces.clone()) {
                        errors.push(format!(
                            "{path}.effects.{effect_name} produces duplicate effect token '{}'",
                            merge.produces
                        ));
                    }
                    effects.insert(effect_name.clone(), merge.produces.clone());
                }

                for (case, (_, _, case_effects, _)) in &case_scopes {
                    for (effect_name, successor) in case_effects {
                        if effects.get(effect_name) != Some(successor)
                            && !merges.contains_key(effect_name)
                        {
                            errors.push(format!(
                                "{path}.cases.{case} changes effect '{effect_name}' without an \
                                 explicit branch merge"
                            ));
                        }
                    }
                }
                if let Some((_, _, default_effects, _)) = &default_scope {
                    for (effect_name, successor) in default_effects {
                        if effects.get(effect_name) != Some(successor)
                            && !merges.contains_key(effect_name)
                        {
                            errors.push(format!(
                                "{path}.default changes effect '{effect_name}' without an explicit \
                                 branch merge"
                            ));
                        }
                    }
                }
            }
            WorkflowNode::Emit {
                value,
                output,
                effect_name,
                effect,
                ..
            } => {
                require_workflow_value(value, values, &format!("{path}.value"), errors);
                if !workflow.outputs.contains_key(output) {
                    errors.push(format!("{path} emits undeclared output '{output}'"));
                } else if let (Some(value_contract), Some(output_contract)) = (
                    value_contracts.get(value),
                    workflow.outputs.get(output).map(|output| &output.contract),
                ) {
                    require_compatible_contracts(
                        value_contract,
                        output_contract,
                        &format!("{path}.output"),
                        errors,
                    );
                }
                apply_effect_transition(
                    effect_name,
                    effect,
                    effects,
                    effect_tokens,
                    &format!("{path}.effect"),
                    errors,
                );
            }
            WorkflowNode::Transfer { input, output, .. } => {
                require_workflow_value(input, values, &format!("{path}.input"), errors);
                define_workflow_value(output, values, &format!("{path}.output"), errors);
                if let Some(contract) = value_contracts.get(input).cloned() {
                    value_contracts.insert(output.clone(), contract);
                }
            }
        }
    }

    fn require_workflow_value(
        value: &str,
        values: &BTreeSet<String>,
        path: &str,
        errors: &mut Vec<String>,
    ) {
        if !values.contains(value) {
            errors.push(format!(
                "{path} references value '{value}' before definition"
            ));
        }
    }

    fn define_workflow_value(
        value: &str,
        values: &mut BTreeSet<String>,
        path: &str,
        errors: &mut Vec<String>,
    ) {
        if !values.insert(value.to_string()) {
            errors.push(format!("{path} redefines SSA value '{value}'"));
        }
    }

    fn require_compatible_contracts(
        source: &crate::schema::TensorContract,
        target: &crate::schema::TensorContract,
        path: &str,
        errors: &mut Vec<String>,
    ) {
        fn normalize_dtype(dtype: &str) -> &str {
            match dtype {
                "fp32" => "float32",
                "fp16" => "float16",
                "bf16" => "bfloat16",
                other => other,
            }
        }
        if normalize_dtype(&source.dtype) != normalize_dtype(&target.dtype)
            || source.rank != target.rank
        {
            errors.push(format!(
                "{path} has incompatible tensor contracts: {} rank {} -> {} rank {}",
                source.dtype, source.rank, target.dtype, target.rank
            ));
            return;
        }
        if let (Some(source_shape), Some(target_shape)) = (&source.shape, &target.shape) {
            for (axis, (source, target)) in source_shape.iter().zip(target_shape).enumerate() {
                if matches!(
                    (source, target),
                    (
                        crate::schema::TensorDimension::Fixed(_),
                        crate::schema::TensorDimension::Fixed(_)
                    )
                ) && source != target
                {
                    errors.push(format!(
                        "{path} has incompatible fixed dimension at axis {axis}"
                    ));
                }
            }
        }
    }

    fn require_state_contract(
        actual: &crate::schema::TensorContract,
        declared: &crate::schema::TensorContract,
        recurrence: &crate::schema::ShapeRecurrence,
        next: bool,
        path: &str,
        errors: &mut Vec<String>,
    ) {
        fn normalize_dtype(dtype: &str) -> &str {
            match dtype {
                "fp32" => "float32",
                "fp16" => "float16",
                "bf16" => "bfloat16",
                other => other,
            }
        }
        if normalize_dtype(&actual.dtype) != normalize_dtype(&declared.dtype)
            || actual.rank != declared.rank
        {
            errors.push(format!(
                "{path} is incompatible with state contract {} rank {}",
                declared.dtype, declared.rank
            ));
            return;
        }
        let (Some(actual_shape), Some(declared_shape)) = (&actual.shape, &declared.shape) else {
            return;
        };
        let growing_axis = match recurrence {
            crate::schema::ShapeRecurrence::Growing { axis, .. } if next => Some(*axis),
            _ => None,
        };
        for (axis, (actual, declared)) in actual_shape.iter().zip(declared_shape).enumerate() {
            if Some(axis) != growing_axis
                && matches!(
                    (actual, declared),
                    (
                        crate::schema::TensorDimension::Fixed(_),
                        crate::schema::TensorDimension::Fixed(_)
                    )
                )
                && actual != declared
            {
                errors.push(format!(
                    "{path} has incompatible state dimension at axis {axis}"
                ));
            }
        }
    }

    fn apply_effect_transition(
        effect: &str,
        transition: &crate::schema::EffectTransition,
        effects: &mut BTreeMap<String, String>,
        tokens: &mut BTreeSet<String>,
        path: &str,
        errors: &mut Vec<String>,
    ) {
        match effects.get(effect) {
            Some(current) if current == &transition.consumes => {}
            Some(current) => errors.push(format!(
                "{path} consumes effect token '{}', but current token is '{current}'",
                transition.consumes
            )),
            None => errors.push(format!("{path} references undeclared effect '{effect}'")),
        }
        if !tokens.insert(transition.produces.clone()) {
            errors.push(format!(
                "{path} produces duplicate effect token '{}'",
                transition.produces
            ));
        }
        effects.insert(effect.to_string(), transition.produces.clone());
    }

    for (name, component) in &spec.models {
        if name.trim().is_empty() {
            errors.push("pipeline model names must not be empty".to_string());
        }

        if name.contains('.') {
            errors.push(format!("pipeline model name must not contain '.': {name}"));
        }
        if component.filename.trim().is_empty() {
            errors.push(format!("pipeline model {name} must declare a filename"));
        }
        if component.role.trim().is_empty() {
            errors.push(format!("pipeline model {name} must declare a type"));
        }
    }
    validate_typed_contracts(spec, &mut errors);
    if let Some(control) = &spec.control {
        if !spec.phases.is_empty() || !legacy_strategy_is_empty(&spec.strategy) {
            errors.push(
                "pipeline.control cannot be combined with legacy pipeline.strategy or \
                 pipeline.phases; express all control flow with pipeline.control"
                    .to_string(),
            );
        }
        validate_control_flow(control, spec, "pipeline.control", &mut errors);
    }

    let mut adjacency: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for name in spec.models.keys() {
        adjacency.entry(name.as_str()).or_default();
    }

    fn legacy_strategy_is_empty(strategy: &PipelineStrategy) -> bool {
        strategy.kind == PipelineStrategyKind::Autoregressive
            && strategy.decoder.is_none()
            && strategy.max_tokens.is_none()
            && strategy.stop_conditions.is_none()
            && strategy.kv_cache.is_none()
            && strategy.speculative.is_none()
            && strategy.model.is_none()
            && strategy.batching.is_none()
            && strategy.denoiser.is_none()
            && strategy.scheduler.is_none()
            && strategy.num_steps.is_none()
            && strategy.timestep_input.is_none()
            && strategy.timesteps.is_none()
            && strategy.start_step.is_none()
            && strategy.scheduler_config.is_none()
            && strategy.cfg_conditioning_input.is_none()
            && strategy.guidance_scale.is_none()
            && strategy.state.is_none()
            && strategy.outer.is_none()
            && strategy.inner.is_none()
            && strategy.num_code_groups.is_none()
            && strategy.pre_embedder.is_none()
            && strategy.inner_embedding_output.is_none()
            && strategy.prefill_embedder.is_none()
            && strategy.stages.is_empty()
    }

    // A same-component self-edge (`A.x -> A.y`) is a legal loop-carried temporal
    // dependency ONLY for the denoiser of an iterative strategy; anywhere else it
    // is treated as a cycle. Collect the components allowed to have self-edges.
    let mut iterative_denoisers: BTreeSet<&str> = BTreeSet::new();
    collect_iterative_denoisers(&spec.strategy, &mut iterative_denoisers);
    if let Some(control) = &spec.control {
        collect_loop_components(control, &mut iterative_denoisers);
    }

    // Each destination port may be fed by at most one edge; multiple producers
    // into one input are ambiguous (order-dependent) and rejected.
    let mut seen_destinations: BTreeSet<&str> = BTreeSet::new();

    for edge in &spec.dataflow {
        match parse_endpoint(&edge.from) {
            Some((component, port)) => {
                if !spec.models.contains_key(component) {
                    errors.push(format!(
                        "dataflow edge source references unknown component: {}",
                        edge.from
                    ));
                }
                if port.is_empty() {
                    errors.push(format!(
                        "dataflow edge source has an empty port: {}",
                        edge.from
                    ));
                }
            }
            None if spec.inputs.contains_key(&edge.from) => {}
            None => errors.push(format!(
                "dataflow edge source must be a declared package input or component.port: {}",
                edge.from
            )),
        }

        match parse_endpoint(&edge.to) {
            Some((component, port)) => {
                if !spec.models.contains_key(component) {
                    errors.push(format!(
                        "dataflow edge destination references unknown component: {}",
                        edge.to
                    ));
                }
                if port.is_empty() {
                    errors.push(format!(
                        "dataflow edge destination has an empty port: {}",
                        edge.to
                    ));
                }
            }
            None if spec.outputs.contains_key(&edge.to) => {}
            None => errors.push(format!(
                "dataflow edge destination must be component.port or a declared package output: {}",
                edge.to
            )),
        }

        if !seen_destinations.insert(edge.to.as_str()) && !spec.reducers.contains_key(&edge.to) {
            errors.push(format!(
                "dataflow has multiple edges into the same destination port: {}",
                edge.to
            ));
        }

        if let (Some((from, _)), Some((to, _))) =
            (parse_endpoint(&edge.from), parse_endpoint(&edge.to))
            && spec.models.contains_key(from)
            && spec.models.contains_key(to)
            // A self-edge is excluded from the acyclic check only when it is an
            // iterative denoiser's loop-carried feedback; a self-edge on any
            // other component is a genuine (unsupported) cycle.
            && !(from == to && iterative_denoisers.contains(from))
        {
            adjacency.entry(from).or_default().insert(to);
        }
    }

    for destination in spec.reducers.keys() {
        let count = spec
            .dataflow
            .iter()
            .filter(|edge| &edge.to == destination)
            .count();
        if count < 2 {
            errors.push(format!(
                "pipeline.reducers.{destination} requires at least two incoming dataflow edges"
            ));
        }
    }

    if spec.control.is_none() {
        let mut strategy_owned = BTreeSet::new();
        collect_strategy_models(&spec.strategy, &mut strategy_owned);

        for phase_component in spec.phases.keys() {
            if !spec.models.contains_key(phase_component) {
                errors.push(format!(
                    "phase references unknown component: {phase_component}"
                ));
            } else if strategy_owned.contains(phase_component.as_str()) {
                errors.push(format!(
                    "pipeline.phases must not contain strategy-owned component {phase_component}; \
                     its lifecycle is defined by pipeline.strategy"
                ));
            }
        }
        for model_component in spec.models.keys() {
            if !strategy_owned.contains(model_component.as_str())
                && !spec.phases.contains_key(model_component)
            {
                errors.push(format!(
                    "auxiliary pipeline model {model_component} must have a pipeline.phases entry \
                     declaring run_on and optional when_present"
                ));
            }
        }
        validate_strategy(&spec.strategy, &spec.models, "strategy", &mut errors);
    }

    validate_acyclic(&adjacency, &mut errors);

    if errors.is_empty() {
        Ok(())
    } else {
        Err(PipelineValidationError { errors })
    }
}

fn validate_typed_contracts(spec: &PipelineSpec, errors: &mut Vec<String>) {
    for (component, model) in &spec.models {
        let ports = &model.ports;
        for (direction, contracts) in [("inputs", &ports.inputs), ("outputs", &ports.outputs)] {
            for (port, contract) in contracts {
                if contract
                    .shape
                    .as_ref()
                    .is_some_and(|shape| shape.len() != contract.rank)
                {
                    errors.push(format!(
                        "pipeline.models.{component}.ports.{direction}.{port} declares rank {} but shape has {} dimensions",
                        contract.rank,
                        contract.shape.as_ref().map_or(0, Vec::len)
                    ));
                }
            }
        }
    }
    for (state, declaration) in &spec.states {
        if declaration
            .contract
            .shape
            .as_ref()
            .is_some_and(|shape| shape.len() != declaration.contract.rank)
        {
            errors.push(format!(
                "pipeline.states.{state}.type rank does not match its shape"
            ));
        }
    }
}

fn validate_control_flow(
    control: &ControlFlow,
    spec: &PipelineSpec,
    path: &str,
    errors: &mut Vec<String>,
) {
    match control {
        ControlFlow::Sequence { steps } => {
            if steps.is_empty() {
                errors.push(format!("{path}.steps must not be empty"));
            }
            for (index, step) in steps.iter().enumerate() {
                validate_control_flow(step, spec, &format!("{path}.steps[{index}]"), errors);
            }
        }
        ControlFlow::Invoke { component, .. } => {
            if !spec.models.contains_key(component) {
                errors.push(format!("{path} invokes unknown component '{component}'"));
            }
        }
        ControlFlow::Loop {
            body,
            carried,
            termination,
            step_program,
        } => {
            if let Termination::Iterations { count, start } = termination
                && (*count == 0 || start > count)
            {
                errors.push(format!(
                    "{path}.termination requires count > 0 and start <= count"
                ));
            }
            for carry in carried {
                if !spec.states.contains_key(&carry.state) {
                    errors.push(format!(
                        "{path}.carried references unknown state '{}'",
                        carry.state
                    ));
                }
                for (field, endpoint) in [("from", &carry.from), ("to", &carry.to)] {
                    match parse_endpoint(endpoint) {
                        Some((component, _)) if spec.models.contains_key(component) => {}
                        _ => errors.push(format!(
                            "{path}.carried.{field} must reference a declared component port, \
                             got '{endpoint}'"
                        )),
                    }
                }
            }
            if let Some(program) = step_program
                && !spec.programs.contains_key(program)
            {
                errors.push(format!(
                    "{path}.step_program references unknown program '{program}'"
                ));
            }
            validate_control_flow(body, spec, &format!("{path}.body"), errors);
        }
        ControlFlow::Branch { cases, default, .. } => {
            if cases.is_empty() && default.is_none() {
                errors.push(format!("{path} must declare a case or default"));
            }
            for (name, case) in cases {
                validate_control_flow(case, spec, &format!("{path}.cases.{name}"), errors);
            }
            if let Some(default) = default {
                validate_control_flow(default, spec, &format!("{path}.default"), errors);
            }
        }
    }
}

fn collect_strategy_models<'a>(strategy: &'a PipelineStrategy, out: &mut BTreeSet<&'a str>) {
    for model in [
        strategy.decoder.as_deref(),
        strategy.model.as_deref(),
        strategy.denoiser.as_deref(),
        strategy.outer.as_deref(),
        strategy.inner.as_deref(),
    ]
    .into_iter()
    .flatten()
    {
        out.insert(model);
    }
    for stage in &strategy.stages {
        collect_strategy_models(&stage.strategy, out);
    }
}

fn parse_endpoint(endpoint: &str) -> Option<(&str, &str)> {
    let (component, port) = endpoint.split_once('.')?;
    if component.is_empty() {
        return None;
    }
    Some((component, port))
}

/// Collect the components that are the `denoiser` of some (possibly nested)
/// iterative strategy — the only components permitted a loop-carried self-edge.
fn collect_iterative_denoisers<'a>(strategy: &'a PipelineStrategy, out: &mut BTreeSet<&'a str>) {
    match strategy.kind {
        PipelineStrategyKind::Iterative => {
            if let Some(denoiser) = strategy.denoiser.as_deref() {
                out.insert(denoiser);
            }
        }
        PipelineStrategyKind::Composite => {
            for stage in &strategy.stages {
                collect_iterative_denoisers(&stage.strategy, out);
            }
        }
        _ => {}
    }
}

fn collect_loop_components<'a>(control: &'a ControlFlow, out: &mut BTreeSet<&'a str>) {
    match control {
        ControlFlow::Sequence { steps } => {
            for step in steps {
                collect_loop_components(step, out);
            }
        }
        ControlFlow::Invoke { .. } => {}
        ControlFlow::Loop { body, .. } => collect_invoked_components(body, out),
        ControlFlow::Branch { cases, default, .. } => {
            for case in cases.values() {
                collect_loop_components(case, out);
            }
            if let Some(default) = default {
                collect_loop_components(default, out);
            }
        }
    }
}

fn collect_invoked_components<'a>(control: &'a ControlFlow, out: &mut BTreeSet<&'a str>) {
    match control {
        ControlFlow::Invoke { component, .. } => {
            out.insert(component);
        }
        ControlFlow::Sequence { steps } => {
            for step in steps {
                collect_invoked_components(step, out);
            }
        }
        ControlFlow::Loop { body, .. } => collect_invoked_components(body, out),
        ControlFlow::Branch { cases, default, .. } => {
            for case in cases.values() {
                collect_invoked_components(case, out);
            }
            if let Some(default) = default {
                collect_invoked_components(default, out);
            }
        }
    }
}

fn validate_strategy(
    strategy: &PipelineStrategy,
    models: &BTreeMap<String, crate::schema::PipelineComponentSpec>,
    path: &str,
    errors: &mut Vec<String>,
) {
    match strategy.kind {
        PipelineStrategyKind::Autoregressive => {
            require_strategy_model(strategy.decoder.as_deref(), "decoder", path, models, errors);
        }
        PipelineStrategyKind::SinglePass => {
            require_strategy_model(strategy.model.as_deref(), "model", path, models, errors);
        }
        PipelineStrategyKind::Iterative => {
            require_strategy_model(
                strategy.denoiser.as_deref(),
                "denoiser",
                path,
                models,
                errors,
            );
            if let (Some(timesteps), Some(num_steps)) =
                (strategy.timesteps.as_ref(), strategy.num_steps)
                && timesteps.len() != num_steps
            {
                errors.push(format!(
                    "{path}.timesteps has {} entries but num_steps is {num_steps}",
                    timesteps.len()
                ));
            }
        }
        PipelineStrategyKind::Composite => {
            if strategy.stages.is_empty() {
                errors.push(format!("{path}.stages must contain at least one stage"));
            }
            for stage in &strategy.stages {
                if stage.name.trim().is_empty() {
                    errors.push(format!("{path}.stages contains a stage with an empty name"));
                }
                validate_strategy(
                    &stage.strategy,
                    models,
                    &format!("{path}.stages[{}]", stage.name),
                    errors,
                );
            }
        }
        PipelineStrategyKind::NestedAutoregressive => {
            require_strategy_model(strategy.outer.as_deref(), "outer", path, models, errors);
            require_strategy_model(strategy.inner.as_deref(), "inner", path, models, errors);
            match strategy.num_code_groups {
                Some(n) if n >= 1 => {}
                Some(_) => errors.push(format!("{path}.num_code_groups must be at least 1")),
                None => errors.push(format!("{path}.num_code_groups is required")),
            }
        }
        PipelineStrategyKind::Other(_) => {}
    }
}

fn require_strategy_model(
    value: Option<&str>,
    field: &str,
    path: &str,
    models: &BTreeMap<String, crate::schema::PipelineComponentSpec>,
    errors: &mut Vec<String>,
) {
    match value {
        Some(name) if models.contains_key(name) => {}
        Some(name) => errors.push(format!(
            "{path}.{field} references unknown component: {name}"
        )),
        None => errors.push(format!("{path}.{field} is required")),
    }
}

fn validate_acyclic(adjacency: &BTreeMap<&str, BTreeSet<&str>>, errors: &mut Vec<String>) {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Mark {
        Visiting,
        Done,
    }

    fn visit<'a>(
        node: &'a str,
        adjacency: &BTreeMap<&'a str, BTreeSet<&'a str>>,
        marks: &mut BTreeMap<&'a str, Mark>,
        stack: &mut Vec<&'a str>,
        errors: &mut Vec<String>,
    ) {
        match marks.get(node) {
            Some(Mark::Done) => return,
            Some(Mark::Visiting) => {
                stack.push(node);
                errors.push(format!(
                    "pipeline dataflow contains a cycle: {}",
                    stack.join(" -> ")
                ));
                stack.pop();
                return;
            }
            None => {}
        }

        marks.insert(node, Mark::Visiting);
        stack.push(node);
        if let Some(next_nodes) = adjacency.get(node) {
            for next in next_nodes {
                visit(next, adjacency, marks, stack, errors);
            }
        }
        stack.pop();
        marks.insert(node, Mark::Done);
    }

    let mut marks = BTreeMap::new();
    let mut stack = Vec::new();
    for node in adjacency.keys() {
        visit(node, adjacency, &mut marks, &mut stack, errors);
    }
}

/// Error type for metadata operations.
#[derive(Debug, thiserror::Error)]
pub enum MetadataError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Unsupported capabilities: {0:?}")]
    Unsupported(Vec<String>),
}

// Re-export at crate level
pub use MetadataError as Error;

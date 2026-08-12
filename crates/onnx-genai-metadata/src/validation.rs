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

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
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
        }
        for (name, state) in &workflow.state {
            if state.scope == crate::schema::WorkflowStateScope::Invocation
                && state.session.is_some()
            {
                errors.push(format!(
                    "workflow state '{name}' has session lease settings but invocation scope"
                ));
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
        let mut effects = workflow.initial_effects.clone();
        let mut effect_tokens = effects.values().cloned().collect::<BTreeSet<_>>();
        validate_workflow_node(
            &workflow.graph,
            workflow,
            &mut values,
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
                let mut body_effects = effects.clone();
                for carry in carried {
                    if !workflow.state.contains_key(&carry.cell) {
                        errors.push(format!(
                            "{path}.carried references unknown cell '{}'",
                            carry.cell
                        ));
                    }
                    require_workflow_value(
                        &carry.current,
                        values,
                        &format!("{path}.carried.current"),
                        errors,
                    );
                    body_values.insert(carry.body_input.clone());
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
                }
                *effects = body_effects;
            }
            WorkflowNode::Branch {
                predicate,
                cases,
                default,
            } => {
                require_workflow_value(predicate, values, &format!("{path}.predicate"), errors);
                if cases.is_empty() {
                    errors.push(format!("{path}.cases must not be empty"));
                }
                let mut resulting_effects = Vec::new();
                for (case, node) in cases {
                    let mut case_values = values.clone();
                    let mut case_effects = effects.clone();
                    validate_workflow_node(
                        node,
                        workflow,
                        &mut case_values,
                        &mut case_effects,
                        effect_tokens,
                        &format!("{path}.cases.{case}"),
                        errors,
                    );
                    resulting_effects.push(case_effects);
                }
                if let Some(default) = default {
                    let mut default_values = values.clone();
                    let mut default_effects = effects.clone();
                    validate_workflow_node(
                        default,
                        workflow,
                        &mut default_values,
                        &mut default_effects,
                        effect_tokens,
                        &format!("{path}.default"),
                        errors,
                    );
                    resulting_effects.push(default_effects);
                }
                if let Some(first) = resulting_effects.first() {
                    if resulting_effects.iter().any(|result| result != first) {
                        errors.push(format!(
                            "{path} has unordered side effects: every branch must produce identical effect tokens"
                        ));
                    } else if let Some(result) = resulting_effects.pop() {
                        *effects = result;
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

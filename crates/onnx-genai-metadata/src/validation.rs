//! Validate metadata against runtime capabilities.

use std::collections::{BTreeMap, BTreeSet};

use crate::capabilities as capability;
use crate::schema::{InferenceMetadata, PipelineSpec, WorkflowNode, WorkflowSpec, WorkflowStep};

struct ContractObligation {
    id: &'static str,
    version: &'static str,
    action: &'static str,
    inputs: &'static [&'static str],
    outputs: &'static [&'static str],
}

const CONTRACT_OBLIGATIONS: &[ContractObligation] = &[
    ContractObligation {
        id: "onnx-genai.grammar-guidance",
        version: "1",
        action: "clone",
        inputs: &["state"],
        outputs: &["next_state"],
    },
    ContractObligation {
        id: "onnx-genai.grammar-guidance",
        version: "1",
        action: "lookahead",
        inputs: &["state", "tokens", "valid_length", "transition_table"],
        outputs: &[
            "next_state",
            "consumed_length",
            "logits_mask",
            "forced_tokens",
            "forced_length",
        ],
    },
    ContractObligation {
        id: "onnx-genai.grammar-guidance",
        version: "1",
        action: "commit",
        inputs: &["state", "tokens", "valid_length", "transition_table"],
        outputs: &["next_state", "consumed_length"],
    },
    ContractObligation {
        id: "onnx-genai.telemetry",
        version: "1",
        action: "start",
        inputs: &[],
        outputs: &["timestamp"],
    },
    ContractObligation {
        id: "onnx-genai.telemetry",
        version: "1",
        action: "elapsed",
        inputs: &["timestamp"],
        outputs: &["duration_ms"],
    },
    ContractObligation {
        id: "onnx-genai.parameter-overlay",
        version: "1",
        action: "apply",
        inputs: &["input"],
        outputs: &["output"],
    },
];

const RETIRED_CONTINUOUS_BATCHING_CAPABILITY: &str = "continuous_batching";

/// Capabilities this runtime supports.
pub struct RuntimeCapabilities {
    pub supported: Vec<String>,
}

impl Default for RuntimeCapabilities {
    fn default() -> Self {
        Self {
            supported: vec![
                capability::KV_CACHE.to_string(),
                capability::GROUPED_QUERY_ATTENTION.to_string(),
                capability::MULTI_HEAD_ATTENTION.to_string(),
                capability::PREFIX_CACHE.to_string(),
                capability::CONTROL_FLOW_LOOP.to_string(),
                // Structural workflow capabilities. Every package now declares
                // its execution as a workflow, including a single decoder, so a
                // runtime that executes workflows implements these by
                // definition. Listing them is what keeps the "declares a
                // capability I do not implement" warning meaningful instead of
                // firing on every load until operators learn to ignore it.
                capability::WORKFLOW_SSA.to_string(),
                capability::LINEAR_EFFECTS.to_string(),
                capability::SERVING_SERVICE_CONTRACT.to_string(),
                capability::NESTED_CONTROL_FLOW.to_string(),
                capability::LOOP_INDUCTION_VALUES.to_string(),
                capability::TYPED_EMIT.to_string(),
                capability::STREAMING_EMIT.to_string(),
            ],
        }
    }
}

/// Validate the metadata document and required runtime capabilities.
///
/// Reports structural defects and unsupported capabilities together. Canonical
/// metadata is an execution contract: callers must reject unsupported required
/// behavior rather than silently selecting a narrower legacy execution path.
pub fn validate(
    metadata: &InferenceMetadata,
    runtime: &RuntimeCapabilities,
) -> Result<(), Vec<String>> {
    let report = validate_structure_and_capabilities(metadata, runtime);
    let mut errors = report.structural;
    errors.extend(report.unsupported_capabilities);
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Structural defects and unsupported capabilities, kept apart.
///
/// These answer different questions while remaining equally fatal at package
/// admission. Keeping them separate lets loaders report whether the package is
/// malformed or the runtime lacks behavior the package requires.
#[derive(Debug, Default, Clone)]
pub struct CapabilityReport {
    /// The document is malformed or self-inconsistent. Always fatal.
    pub structural: Vec<String>,
    /// Capabilities the package declares that this runtime does not implement.
    pub unsupported_capabilities: Vec<String>,
}

/// Validate the document, reporting structural defects separately from
/// capabilities this runtime does not implement.
pub fn validate_structure_and_capabilities(
    metadata: &InferenceMetadata,
    runtime: &RuntimeCapabilities,
) -> CapabilityReport {
    let structural = validate_metadata(metadata).err().unwrap_or_default();
    let required = metadata
        .required_capabilities
        .iter()
        .cloned()
        .chain(derived_capabilities(metadata))
        .collect::<BTreeSet<_>>();
    let unsupported_capabilities = required
        .into_iter()
        .filter(|capability| !runtime.supported.contains(capability))
        .collect();
    CapabilityReport {
        structural,
        unsupported_capabilities,
    }
}

fn metadata_only_required_capabilities(metadata: &InferenceMetadata) -> BTreeSet<String> {
    let mut capabilities = BTreeSet::new();
    if metadata.adapters.is_some() {
        capabilities.insert(capability::PARAMETER_ADAPTERS.to_string());
        capabilities.insert(capability::HETEROGENEOUS_ADAPTER_BATCHING.to_string());
    }
    capabilities
}

fn workflow_required_capabilities(
    workflow: &WorkflowSpec,
    compiled: Option<&WorkflowNode>,
) -> BTreeSet<String> {
    let mut capabilities = BTreeSet::new();
    capabilities.insert(capability::WORKFLOW_SSA.to_string());
    if workflow.serving.is_some() {
        capabilities.insert(capability::SERVING_SERVICE_CONTRACT.to_string());
    }
    if workflow
        .state
        .values()
        .any(|state| state.scope == crate::schema::WorkflowStateScope::Session)
    {
        capabilities.insert(capability::SESSION_STATE_LEASE.to_string());
    }
    if workflow.state.values().any(|state| {
        matches!(
            state.recurrence,
            crate::schema::ShapeRecurrence::Bounded { .. }
        )
    }) {
        capabilities.insert(capability::BOUNDED_STATE_RECURRENCE.to_string());
    }
    if workflow
        .state
        .values()
        .any(|state| state.class == crate::schema::WorkflowStateClass::Advisory)
    {
        capabilities.insert(capability::ADVISORY_STATE.to_string());
    }
    if workflow
        .inputs
        .values()
        .any(|input| input.present_as.is_some())
    {
        capabilities.insert(capability::INPUT_PRESENCE.to_string());
    }
    if workflow.inputs.values().any(|input| {
        matches!(
            &input.role,
            crate::schema::SemanticInputRole::Runtime {
                role: crate::schema::RuntimeInputRole::AdapterSegments
                    | crate::schema::RuntimeInputRole::AdapterCounts
                    | crate::schema::RuntimeInputRole::AdapterScales
                    | crate::schema::RuntimeInputRole::AdapterActive,
                ..
            }
        )
    }) {
        capabilities.insert(capability::HETEROGENEOUS_ADAPTER_BATCHING.to_string());
    }
    if !workflow.effects.is_empty()
        || workflow
            .components
            .values()
            .any(|component| !component.effects.is_empty())
    {
        capabilities.insert(capability::LINEAR_EFFECTS.to_string());
    }
    for component in workflow.components.values() {
        let contract_id = component
            .contract
            .as_ref()
            .map(|contract| contract.id.as_str());
        let adapter_abi = match &component.implementation {
            crate::schema::ComponentImplementation::Adapter { abi, .. } => Some(abi.as_str()),
            _ => None,
        };
        for identifier in contract_id.into_iter().chain(adapter_abi) {
            match identifier {
                "onnx-genai.adaptive-proposal-budget" => {
                    capabilities.insert(capability::ADAPTIVE_PROPOSAL_BUDGET.to_string());
                }
                "onnx-genai.grammar-guidance" => {
                    capabilities.insert(capability::GRAMMAR_GUIDANCE_ADAPTER.to_string());
                }
                "onnx-genai.telemetry" => {
                    capabilities.insert(capability::TELEMETRY_ADAPTER.to_string());
                }
                "onnx-genai.parameter-overlay" => {
                    capabilities.insert(capability::PARAMETER_ADAPTERS.to_string());
                }
                _ => {}
            }
        }
    }
    if let Some(compiled) = compiled {
        collect_workflow_capabilities(compiled, &mut capabilities);
    }
    capabilities
}

fn metadata_required_capabilities(metadata: &InferenceMetadata) -> BTreeSet<String> {
    let mut capabilities = metadata_only_required_capabilities(metadata);
    if let Some(pipeline) = &metadata.pipeline {
        let compiled = crate::compile_workflow(&pipeline.workflow).ok();
        capabilities.extend(workflow_required_capabilities(
            &pipeline.workflow,
            compiled.as_ref().map(|compiled| &compiled.graph),
        ));
    }
    capabilities
}

/// Capabilities implied by concrete metadata features.
pub fn derived_capabilities(metadata: &InferenceMetadata) -> BTreeSet<String> {
    let mut capabilities = metadata_required_capabilities(metadata);
    if let Some(pipeline) = &metadata.pipeline {
        capabilities.extend(pipeline.workflow.manifest.capabilities.iter().cloned());
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
                capabilities.insert(capability::LINEAR_EFFECTS.to_string());
            }
        }
        WorkflowNode::Loop {
            setup,
            body,
            iteration,
            effects,
            ..
        } => {
            capabilities.insert(capability::NESTED_CONTROL_FLOW.to_string());
            if iteration.is_some() {
                capabilities.insert(capability::LOOP_INDUCTION_VALUES.to_string());
            }
            if !effects.is_empty() {
                capabilities.insert(capability::LINEAR_EFFECTS.to_string());
            }
            collect_workflow_capabilities(setup, capabilities);
            collect_workflow_capabilities(body, capabilities);
        }
        WorkflowNode::Branch {
            cases,
            default,
            effects,
            ..
        } => {
            capabilities.insert(capability::NESTED_CONTROL_FLOW.to_string());
            if !effects.is_empty() {
                capabilities.insert(capability::LINEAR_EFFECTS.to_string());
            }
            for case in cases.values() {
                collect_workflow_capabilities(case, capabilities);
            }
            if let Some(default) = default {
                collect_workflow_capabilities(default, capabilities);
            }
        }
        WorkflowNode::Emit {
            mode, valid_length, ..
        } => {
            capabilities.insert(capability::TYPED_EMIT.to_string());
            if matches!(mode, crate::schema::WorkflowEmitMode::Event) {
                capabilities.insert(capability::STREAMING_EMIT.to_string());
            }
            if valid_length.is_some() {
                capabilities.insert(capability::EMIT_VALID_LENGTH.to_string());
            }
        }
        WorkflowNode::Transfer { .. } => {
            capabilities.insert(capability::EXPLICIT_TRANSFER.to_string());
        }
        WorkflowNode::ExecutionIsland { .. } => {}
    }
}

/// Validate document-level invariants independent of runtime capabilities.
pub fn validate_metadata(metadata: &InferenceMetadata) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    // An unreadable version is reported once, by `validate_schema_version`
    // below. Falling back to the initial version here validates the rest of the
    // document at the most permissive strictness rather than adding a second,
    // derived complaint about the same missing fact.
    let version = crate::version::normalize(metadata.schema_version.as_deref())
        .unwrap_or(crate::version::INITIAL_SCHEMA_VERSION);
    if let Some(pipeline) = &metadata.pipeline
        && let Err(error) = validate_pipeline_spec(pipeline, version)
    {
        errors.extend(error.errors);
    }
    if let Some(service) = &metadata.adapters {
        validate_adapter_service(
            service,
            metadata.pipeline.as_ref().map(|p| &p.workflow),
            &mut errors,
        );
        if let Some(workflow) = metadata
            .pipeline
            .as_ref()
            .map(|pipeline| &pipeline.workflow)
        {
            for capability in metadata_only_required_capabilities(metadata)
                .difference(&workflow.manifest.capabilities)
            {
                errors.push(format!(
                    "pipeline.workflow.manifest.capabilities is missing used capability '{capability}'"
                ));
            }
        }
    }
    validate_schema_version(metadata, &mut errors);
    validate_retired_batching_capability(metadata, &mut errors);
    validate_preprocessing_workflow(metadata, &mut errors);
    validate_token_authority(metadata, version, &mut errors);
    validate_generation_contract(metadata, &mut errors);
    validate_profiles(metadata, &mut errors);
    validate_profile_decoding(metadata, &mut errors);
    validate_speculative_rollback(metadata, &mut errors);

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// A document declares the version whose fields it actually uses.
///
/// Absence is the compatibility mechanism in a schema that denies unknown
/// fields: a package that uses no *added* field keeps loading on every runtime
/// it loaded on before, and one that uses something new needs a reader that
/// knows it. That only works if the declared version tells the truth, so a
/// document carrying a `1.1` field while claiming `1.0` is refused here rather
/// than discovered by an older runtime as a mystery unknown-field error.
///
/// The emphasis is load-bearing: absence buys nothing for a field that was
/// *reshaped* rather than added, because the old spelling is absent from the
/// new reader by construction. `token_packed` is this schema's one such case
/// while it is pre-release, and it is refused by name in
/// [`crate::parser`] instead of being read as a version mismatch it is not.
fn validate_schema_version(metadata: &InferenceMetadata, errors: &mut Vec<String>) {
    let declared = match crate::version::normalize(metadata.schema_version.as_deref()) {
        Ok(version) => version,
        Err(error) => {
            errors.push(error);
            return;
        }
    };
    if declared > crate::version::SUPPORTED_SCHEMA_VERSION {
        errors.push(format!(
            "this package declares inference-metadata schema version {declared}, and this build \
             reads up to {}",
            crate::version::SUPPORTED_SCHEMA_VERSION
        ));
        return;
    }
    if metadata.tokens.is_some() && declared < crate::version::TOKEN_AUTHORITY_SCHEMA_VERSION {
        let spelled = metadata.schema_version.as_deref().unwrap_or("<absent>");
        errors.push(format!(
            "this package declares top-level `tokens`, which schema version {} introduced, but \
             declares schema_version '{spelled}' ({declared}); declare schema_version '{}' so an \
             older reader refuses the package as a newer contract rather than reporting \
             `tokens` as an unknown field",
            crate::version::TOKEN_AUTHORITY_SCHEMA_VERSION,
            crate::version::TOKEN_AUTHORITY_SCHEMA_VERSION
        ));
    }
    let Some(feature) = batching_schema_feature(metadata) else {
        return;
    };
    let required = crate::version::BATCHING_SCHEMA_VERSION;
    if declared < required {
        let spelled = metadata.schema_version.as_deref().unwrap_or("<absent>");
        errors.push(format!(
            "this package {feature}, which schema version {required} introduced, but declares \
             schema_version '{spelled}' ({declared}). Every structure in this schema refuses \
             fields it does not know, so a reader built for {declared} would reject this document \
             with a puzzled unknown-field error; declare schema_version '{required}' so it is \
             refused for the reason that is true"
        ));
    }
}

/// Enforce one authority for numeric token facts and one executable stop policy.
fn validate_token_authority(
    metadata: &InferenceMetadata,
    version: crate::version::SchemaVersion,
    errors: &mut Vec<String>,
) {
    if let Some(tokens) = &metadata.tokens {
        let mut unique = BTreeSet::new();
        for id in &tokens.eos_token_id {
            if !unique.insert(id) {
                errors.push(format!(
                    "tokens.eos_token_id repeats id {id}; EOS ids are an ordered set"
                ));
            }
        }
    }

    let Some(workflow) = metadata
        .pipeline
        .as_ref()
        .map(|pipeline| &pipeline.workflow)
    else {
        return;
    };

    for (name, input) in &workflow.inputs {
        if matches!(name.as_str(), "package.eos_ids" | "package.eos_token_ids") {
            errors.push(format!(
                "workflow input '{name}' is a retired duplicate of package token facts; move its \
                     numeric ids to top-level `tokens.eos_token_id` and let an optional request \
                     `eos_token_ids` role override them"
            ));
        }
        if matches!(
            &input.role,
            crate::schema::SemanticInputRole::Runtime {
                role: crate::schema::RuntimeInputRole::EosTokenIds
                    | crate::schema::RuntimeInputRole::EosTokenLengths,
                ..
            }
        ) && input.default.is_some()
        {
            errors.push(format!(
                "workflow input '{name}' is a request EOS override but declares a literal \
                     default; remove the default because package defaults derive from \
                     top-level `tokens.eos_token_id`"
            ));
        }
    }

    if version < crate::version::TOKEN_AUTHORITY_SCHEMA_VERSION {
        return;
    }

    let has_eos = metadata
        .tokens
        .as_ref()
        .is_some_and(|tokens| !tokens.eos_token_id.is_empty());
    validate_generation_eos_steps(&workflow.steps, workflow, has_eos, errors);

    if metadata.speculative.is_some() {
        if !has_eos {
            errors.push(
                "a speculative autoregressive package must declare non-empty \
                     `tokens.eos_token_id`"
                    .to_string(),
            );
        }
        if !workflow.components.values().any(|component| {
            component
                .contract
                .as_ref()
                .is_some_and(is_termination_contract)
        }) {
            errors.push(
                "a speculative autoregressive package must declare a component \
                     implementing `onnx-genai.termination-predicate` or `onnx-genai.token-policy`; \
                     this may be a portable ONNX policy graph or an explicit runtime binding, but the \
                     speculative descriptor alone does not state stop semantics"
                        .to_string(),
            );
        }
    }
}

fn validate_generation_eos_steps(
    steps: &[WorkflowStep],
    workflow: &WorkflowSpec,
    has_eos: bool,
    errors: &mut Vec<String>,
) {
    for step in steps {
        match step {
            WorkflowStep::Sequence { steps } => {
                validate_generation_eos_steps(steps, workflow, has_eos, errors);
            }
            WorkflowStep::Loop {
                setup,
                steps,
                termination,
                ..
            } => {
                if *termination == crate::schema::WorkflowLoopTermination::GenerationEos {
                    if !has_eos {
                        errors.push(
                            "a `generation_eos` workflow loop must declare non-empty \
                                 top-level `tokens.eos_token_id`"
                                .to_string(),
                        );
                    }
                    let mut invoked = BTreeSet::new();
                    collect_token_policy_invocations(setup, &mut invoked);
                    collect_token_policy_invocations(steps, &mut invoked);
                    let has_policy = invoked.iter().any(|name| {
                        workflow
                            .components
                            .get(*name)
                            .and_then(|component| component.contract.as_ref())
                            .is_some_and(is_termination_contract)
                    });
                    if !has_policy {
                        errors.push(
                                "a `generation_eos` workflow loop must invoke a component declaring \
                                 `onnx-genai.termination-predicate` or `onnx-genai.token-policy`; EOS \
                                 is executable semantics, not an implicit model-family behavior"
                                    .to_string(),
                            );
                    }
                }
                validate_generation_eos_steps(setup, workflow, has_eos, errors);
                validate_generation_eos_steps(steps, workflow, has_eos, errors);
            }
            WorkflowStep::Branch { cases, default, .. } => {
                for case in cases.values() {
                    validate_generation_eos_steps(
                        std::slice::from_ref(case),
                        workflow,
                        has_eos,
                        errors,
                    );
                }
                if let Some(default) = default {
                    validate_generation_eos_steps(
                        std::slice::from_ref(default),
                        workflow,
                        has_eos,
                        errors,
                    );
                }
            }
            WorkflowStep::Invoke { .. } | WorkflowStep::Emit { .. } => {}
        }
    }
}

fn collect_token_policy_invocations<'a>(
    steps: &'a [WorkflowStep],
    invoked: &mut BTreeSet<&'a str>,
) {
    for step in steps {
        match step {
            WorkflowStep::Invoke { component, .. } => {
                invoked.insert(component);
            }
            WorkflowStep::Sequence { steps } => collect_token_policy_invocations(steps, invoked),
            WorkflowStep::Loop { setup, steps, .. } => {
                collect_token_policy_invocations(setup, invoked);
                collect_token_policy_invocations(steps, invoked);
            }
            WorkflowStep::Branch { cases, default, .. } => {
                for case in cases.values() {
                    collect_token_policy_invocations(std::slice::from_ref(case), invoked);
                }
                if let Some(default) = default {
                    collect_token_policy_invocations(std::slice::from_ref(default), invoked);
                }
            }
            WorkflowStep::Emit { .. } => {}
        }
    }
}

fn is_termination_contract(contract: &crate::schema::ComponentContract) -> bool {
    matches!(
        contract.id.as_str(),
        "onnx-genai.termination-predicate" | "onnx-genai.token-policy"
    )
}

/// Reject the old opt-in spelling for an optimization the runtime derives.
///
/// A capability says execution would be incorrect without a behavior.
/// Continuous batching is never required for correctness: a runtime may always
/// execute one request at a time. Whether a resolved decode path can share a
/// forward is derived from its graph/state contract and backend, while whether
/// to do so is deployment policy.
fn validate_retired_batching_capability(metadata: &InferenceMetadata, errors: &mut Vec<String>) {
    if metadata
        .required_capabilities
        .iter()
        .any(|capability| capability == RETIRED_CONTINUOUS_BATCHING_CAPABILITY)
    {
        errors.push(
            "required_capabilities contains retired capability 'continuous_batching'; remove it. \
             Shared-forward support is derived from the workflow and resolved backend, and \
             enabling or sizing batches is runtime policy, so single-request execution remains \
             correct without a negotiated capability"
                .to_string(),
        );
    }
    if let Some(workflow) = metadata
        .pipeline
        .as_ref()
        .map(|pipeline| &pipeline.workflow)
        && workflow
            .manifest
            .capabilities
            .iter()
            .any(|capability| capability == RETIRED_CONTINUOUS_BATCHING_CAPABILITY)
    {
        errors.push(
            "pipeline.workflow.manifest.capabilities contains retired capability \
             'continuous_batching'; remove it. The workflow's typed batch layouts, state groups, \
             and row-scoped ABI are the structural contract; grouping remains a runtime choice"
                .to_string(),
        );
    }
}

/// The first `1.1` field this document uses, described the way a document
/// writer would recognize it.
fn batching_schema_feature(metadata: &InferenceMetadata) -> Option<String> {
    let mut sites: Vec<(String, &crate::schema::TensorContract)> = Vec::new();
    if let Some(preprocessing) = &metadata.preprocessing {
        if preprocessing.video.is_some() {
            return Some("declares preprocessing.video".to_string());
        }
        if let Some(program) = &preprocessing.image {
            for binding in &program.outputs {
                if let Some(contract) = &binding.contract {
                    sites.push((
                        format!("preprocessing.image output '{}'", binding.name),
                        contract,
                    ));
                }
            }
        }
        if let Some(program) = &preprocessing.audio {
            for binding in &program.outputs {
                if let Some(contract) = &binding.contract {
                    sites.push((
                        format!("preprocessing.audio output '{}'", binding.name),
                        contract,
                    ));
                }
            }
        }
    }
    if let Some(workflow) = metadata
        .pipeline
        .as_ref()
        .map(|pipeline| &pipeline.workflow)
    {
        for (name, component) in &workflow.components {
            if component.batch_capacity.is_some() {
                return Some(format!(
                    "declares batch_capacity on workflow component '{name}'"
                ));
            }
        }
        for (name, input) in &workflow.inputs {
            sites.push((format!("workflow input '{name}'"), &input.contract));
        }
        for (name, output) in &workflow.outputs {
            sites.push((format!("workflow output '{name}'"), &output.contract));
        }
        for (name, state) in &workflow.state {
            sites.push((format!("workflow state '{name}'"), &state.contract));
        }
        for (component, spec) in &workflow.components {
            for (direction, ports) in [
                ("input", &spec.ports.inputs),
                ("output", &spec.ports.outputs),
            ] {
                for (port, contract) in ports {
                    sites.push((
                        format!("workflow component '{component}' {direction} '{port}'"),
                        contract,
                    ));
                }
            }
        }
    }
    for (path, contract) in sites {
        if !contract.padding.is_empty() {
            return Some(format!("declares padding on {path}"));
        }
        if !contract.batch_layout.levels().is_empty() {
            return Some(format!("declares a token_packed ownership chain on {path}"));
        }
    }
    None
}

/// Generation overrides must be structural: every overridable field binds a
/// request-sourced typed workflow input. A caller override of anything else has
/// no representation and must fail loudly instead of being silently dropped.
/// Components that execute inside the speculative region.
///
/// The region is the innermost loop body that invokes both the proposer and the
/// target, because that is the body re-run per speculated position. When the two
/// are not invoked inside a common loop there is no iterated region, and only
/// the named components are constrained.
fn speculative_region_components(
    steps: &[WorkflowStep],
    proposer: &str,
    target: &str,
) -> BTreeSet<String> {
    let mut region = BTreeSet::new();
    for step in steps {
        if find_speculative_region(step, proposer, target, &mut region) {
            return region;
        }
    }
    region.insert(proposer.to_string());
    region.insert(target.to_string());
    region
}

/// Walk for the innermost loop invoking both roles, collecting its components.
///
/// Returns true once the region has been found so the caller stops descending.
fn find_speculative_region(
    step: &WorkflowStep,
    proposer: &str,
    target: &str,
    region: &mut BTreeSet<String>,
) -> bool {
    match step {
        WorkflowStep::Loop { setup, steps, .. } => {
            // Prefer the innermost enclosing loop: a nested loop that still
            // invokes both roles is the tighter region.
            for nested in steps.iter().chain(setup) {
                if find_speculative_region(nested, proposer, target, region) {
                    return true;
                }
            }
            let mut invoked = BTreeSet::new();
            for nested in steps.iter().chain(setup) {
                collect_invoked_components(nested, &mut invoked);
            }
            if invoked.contains(proposer) && invoked.contains(target) {
                region.extend(invoked);
                return true;
            }
            false
        }
        WorkflowStep::Sequence { steps } => steps
            .iter()
            .any(|nested| find_speculative_region(nested, proposer, target, region)),
        WorkflowStep::Branch { cases, default, .. } => cases
            .values()
            .chain(default.iter().map(AsRef::as_ref))
            .any(|nested| find_speculative_region(nested, proposer, target, region)),
        WorkflowStep::Invoke { .. } | WorkflowStep::Emit { .. } => false,
    }
}

/// Every component invoked anywhere beneath a step.
fn collect_invoked_components(step: &WorkflowStep, invoked: &mut BTreeSet<String>) {
    match step {
        WorkflowStep::Invoke { component, .. } => {
            invoked.insert(component.clone());
        }
        WorkflowStep::Sequence { steps } => {
            for nested in steps {
                collect_invoked_components(nested, invoked);
            }
        }
        WorkflowStep::Loop { setup, steps, .. } => {
            for nested in steps.iter().chain(setup) {
                collect_invoked_components(nested, invoked);
            }
        }
        WorkflowStep::Branch { cases, default, .. } => {
            for nested in cases.values().chain(default.iter().map(AsRef::as_ref)) {
                collect_invoked_components(nested, invoked);
            }
        }
        WorkflowStep::Emit { .. } => {}
    }
}

/// Workflow state cells whose value leaves the package through an `emit`.
///
/// An emit names an SSA value and an output key that need not match, so a state
/// cell can reach a package output under an entirely different name. Publication
/// has to be detected on the emitted value, never on the output key.
fn emitted_state_cells(steps: &[WorkflowStep], state: &BTreeSet<&str>) -> BTreeSet<String> {
    let mut emitted = BTreeSet::new();
    for step in steps {
        collect_emitted_state_cells(step, state, &mut emitted);
    }
    emitted
}

fn collect_emitted_state_cells(
    step: &WorkflowStep,
    state: &BTreeSet<&str>,
    emitted: &mut BTreeSet<String>,
) {
    match step {
        WorkflowStep::Emit { value, .. } => {
            if state.contains(value.as_str()) {
                emitted.insert(value.clone());
            }
        }
        WorkflowStep::Sequence { steps } => {
            for nested in steps {
                collect_emitted_state_cells(nested, state, emitted);
            }
        }
        WorkflowStep::Loop { setup, steps, .. } => {
            // A loop republishes each carried cell into the enclosing scope
            // under the cell's own name, so an emit of that alias is already
            // caught by the Emit arm above.
            for nested in steps.iter().chain(setup) {
                collect_emitted_state_cells(nested, state, emitted);
            }
        }
        WorkflowStep::Branch { cases, default, .. } => {
            for nested in cases.values().chain(default.iter().map(AsRef::as_ref)) {
                collect_emitted_state_cells(nested, state, emitted);
            }
        }
        WorkflowStep::Invoke { .. } => {}
    }
}

fn validate_generation_contract(metadata: &InferenceMetadata, errors: &mut Vec<String>) {
    let Some(generation) = &metadata.generation else {
        return;
    };
    let workflow = metadata
        .pipeline
        .as_ref()
        .map(|pipeline| &pipeline.workflow);
    for (field, over) in &generation.overrides {
        if field.trim().is_empty() {
            errors.push("generation.overrides contains an empty field name".to_string());
        }
        let Some(workflow) = workflow else {
            errors.push(format!(
                "generation.overrides.{field} requires pipeline.workflow to declare the \
                 request-sourced input '{}'",
                over.input
            ));
            continue;
        };
        match workflow.inputs.get(&over.input) {
            Some(input)
                if matches!(input.source, crate::schema::WorkflowInputSource::Request)
                    && matches!(input.role, crate::schema::SemanticInputRole::Runtime { .. }) => {}
            Some(_) => errors.push(format!(
                "generation.overrides.{field} input '{}' must be a request-sourced workflow input \
                 with a versioned runtime role",
                over.input
            )),
            None => errors.push(format!(
                "generation.overrides.{field} references undeclared workflow input '{}'",
                over.input
            )),
        }
        if let Some(constraint) = &over.constraint
            && let (Some(minimum), Some(maximum)) = (constraint.minimum, constraint.maximum)
            && minimum > maximum
        {
            errors.push(format!(
                "generation.overrides.{field} constraint minimum {minimum} exceeds maximum \
                 {maximum}"
            ));
        }
    }
}

/// Profile kinds this reader can interpret. An unrecognized kind is only
/// executable by a reader that understands it; here it is either a hard
/// error (`requirement: required`) or fully skipped (`requirement:
/// ignorable`), including skipping its `decoding` block in
/// `validate_profile_decoding`.
const KNOWN_PROFILE_KINDS: &[&str] = &[
    "generation",
    "embedding",
    "reranking",
    "classification",
    "reward",
    "transcription",
];

/// A required profile must be executable by this reader; an ignorable profile
/// may be skipped. Unknown core fields still fail through `deny_unknown_fields`.
fn validate_profiles(metadata: &InferenceMetadata, errors: &mut Vec<String>) {
    let workflow = metadata
        .pipeline
        .as_ref()
        .map(|pipeline| &pipeline.workflow);
    for (name, profile) in &metadata.profiles {
        let known = KNOWN_PROFILE_KINDS.contains(&profile.kind.as_str());
        if !known && profile.requirement == crate::schema::ProfileRequirement::Required {
            errors.push(format!(
                "profiles.{name} declares required kind '{}', which this reader cannot execute; \
                 declare requirement 'ignorable' to let strict readers skip it",
                profile.kind
            ));
            continue;
        }
        if !known {
            continue;
        }
        for (role, output) in &profile.outputs {
            match workflow {
                Some(workflow) if workflow.outputs.contains_key(output) => {}
                Some(_) => errors.push(format!(
                    "profiles.{name}.outputs.{role} references undeclared workflow output \
                     '{output}'"
                )),
                None => errors.push(format!(
                    "profiles.{name}.outputs.{role} requires pipeline.workflow to declare output \
                     '{output}'"
                )),
            }
        }
    }
}

/// A CTC (or other frame-synchronous) profile declares how its raw sequence
/// output becomes discrete tokens. These rules only apply to a `kind` this
/// reader recognizes: an unknown, ignorable profile is fully skippable, so its
/// `decoding` block (if any) is never interpreted either.
fn validate_profile_decoding(metadata: &InferenceMetadata, errors: &mut Vec<String>) {
    for (name, profile) in &metadata.profiles {
        if !KNOWN_PROFILE_KINDS.contains(&profile.kind.as_str()) {
            continue;
        }
        if profile.kind == "transcription" && profile.decoding.is_none() {
            errors.push(format!(
                "profiles.{name}.decoding is required because profiles.{name}.kind is \
                 'transcription'"
            ));
        }
        let Some(decoding) = &profile.decoding else {
            continue;
        };
        if decoding.kind == "ctc" && decoding.blank_id.is_none() {
            errors.push(format!(
                "profiles.{name}.decoding requires blank_id because kind is 'ctc'"
            ));
        }
        if decoding.time_axis == decoding.class_axis {
            errors.push(format!(
                "profiles.{name}.decoding.time_axis and class_axis must not both be axis {}",
                decoding.time_axis
            ));
        }
        if let Some(role) = &decoding.lengths
            && !profile.outputs.contains_key(role)
        {
            errors.push(format!(
                "profiles.{name}.decoding references output role '{role}' that the profile does \
                 not declare"
            ));
        }
        if decoding.kind == "ctc" {
            validate_ctc_logits_contract(metadata, name, profile, decoding, errors);
        }
        if let Some(vocabulary) = &decoding.vocabulary {
            if vocabulary.source == "inline" && vocabulary.tokens.is_empty() {
                errors.push(format!(
                    "profiles.{name}.decoding.vocabulary requires non-empty tokens because \
                     source is 'inline'"
                ));
            }
            if let Some(size) = vocabulary.size
                && !vocabulary.tokens.is_empty()
                && size != vocabulary.tokens.len()
            {
                errors.push(format!(
                    "profiles.{name}.decoding.vocabulary size {size} disagrees with tokens \
                     length {}",
                    vocabulary.tokens.len()
                ));
            }
            // A renderer resolves the delimiter and the ignore list by string
            // identity against the inline table, so a token that is absent
            // there can never match and would silently change the transcript.
            if !vocabulary.tokens.is_empty() {
                if let Some(delimiter) = &vocabulary.word_delimiter
                    && !vocabulary.tokens.contains(delimiter)
                {
                    errors.push(format!(
                        "profiles.{name}.decoding.vocabulary.word_delimiter '{delimiter}' is \
                         not present in tokens"
                    ));
                }
                for ignored in &vocabulary.ignored_tokens {
                    if !vocabulary.tokens.contains(ignored) {
                        errors.push(format!(
                            "profiles.{name}.decoding.vocabulary.ignored_tokens entry \
                             '{ignored}' is not present in tokens"
                        ));
                    }
                }
                if let Some(blank_id) = decoding.blank_id
                    && blank_id as usize >= vocabulary.tokens.len()
                {
                    errors.push(format!(
                        "profiles.{name}.decoding.blank_id {blank_id} is out of range for a \
                         vocabulary of {} tokens",
                        vocabulary.tokens.len()
                    ));
                }
            }
        }
    }
}

/// Resolve CTC through its canonical logits role and bind padded time logits
/// to the padding contract's one validity truth.
///
/// `TensorContract::padding` already validates the companion's int64 dtype,
/// shared layout, and exact outer-axis rank/shape. This cross-layer rule makes
/// the CTC decoder consume that same value by name rather than permitting a
/// missing or second, contradictory frame-count source.
fn validate_ctc_logits_contract(
    metadata: &InferenceMetadata,
    profile_name: &str,
    profile: &crate::schema::TaskProfile,
    decoding: &crate::schema::SequenceDecodingSpec,
    errors: &mut Vec<String>,
) {
    let Some(logits_output) = profile.outputs.get("logits") else {
        errors.push(format!(
            "profiles.{profile_name}.outputs.logits is required because CTC decoding reads the \
             canonical 'logits' role; map that role to the workflow output containing the \
             frame-by-class logits tensor"
        ));
        return;
    };
    let Some(workflow) = metadata
        .pipeline
        .as_ref()
        .map(|pipeline| &pipeline.workflow)
    else {
        return;
    };
    let Some(logits) = workflow.outputs.get(logits_output) else {
        return;
    };
    if decoding.time_axis >= logits.contract.rank {
        errors.push(format!(
            "profiles.{profile_name}.decoding.time_axis {} is outside workflow output \
             '{logits_output}' rank {}",
            decoding.time_axis, logits.contract.rank
        ));
        return;
    }
    if decoding.class_axis >= logits.contract.rank {
        errors.push(format!(
            "profiles.{profile_name}.decoding.class_axis {} is outside workflow output \
             '{logits_output}' rank {}",
            decoding.class_axis, logits.contract.rank
        ));
    }
    let Some(padding) = logits.contract.padding.iter().find(|padding| {
        axis_of_symbol(&logits.contract, &padding.dimension) == Some(decoding.time_axis)
    }) else {
        return;
    };
    let Some(lengths_role) = decoding.lengths.as_deref() else {
        errors.push(format!(
            "profiles.{profile_name}.decoding.lengths is required because workflow output \
             '{logits_output}' pads decoded time axis {} ('{}') with valid_lengths '{}'; CTC \
             must decode exactly that valid prefix",
            decoding.time_axis, padding.dimension, padding.valid_lengths
        ));
        return;
    };
    let Some(bound_output) = profile.outputs.get(lengths_role) else {
        return;
    };
    if bound_output != &padding.valid_lengths {
        errors.push(format!(
            "profiles.{profile_name}.decoding.lengths role '{lengths_role}' binds workflow output \
             '{bound_output}', but workflow output '{logits_output}' pads decoded time axis {} \
             ('{}') with valid_lengths '{}'; bind decoding.lengths to a profile output role \
             mapping to '{}' so padding and CTC decoding have one source of truth",
            decoding.time_axis, padding.dimension, padding.valid_lengths, padding.valid_lengths
        ));
    }
}

/// One preprocessing output binding, viewed independently of its modality.
///
/// Image, video, and audio programs bind processor-local values to typed SSA
/// names under identical rules, so the workflow checks below are written once
/// against this view rather than duplicated per modality.
struct PreprocessingOutputView<'a> {
    name: &'a str,
    dtype: &'a str,
    content: &'a str,
    contract: Option<&'a crate::schema::TensorContract>,
    optional: bool,
}

impl<'a> PreprocessingOutputView<'a> {
    fn pixels(program: &'a crate::schema::VisionPreprocessingProgram) -> Vec<Self> {
        program
            .outputs
            .iter()
            .map(|output| PreprocessingOutputView {
                name: &output.name,
                dtype: &output.dtype,
                content: &output.content,
                contract: output.contract.as_ref(),
                optional: output.optional.unwrap_or(false),
            })
            .collect()
    }

    fn audio(program: &'a crate::schema::AudioPreprocessingProgram) -> Vec<Self> {
        program
            .outputs
            .iter()
            .map(|output| PreprocessingOutputView {
                name: &output.name,
                dtype: &output.dtype,
                content: &output.content,
                contract: output.contract.as_ref(),
                optional: output.optional.unwrap_or(false),
            })
            .collect()
    }
}

/// A companion a program emits carries the role that says what it is.
///
/// The reference in an ownership level says *that* a value describes a packed
/// tensor; the content role says *what a runtime may do with it*. There are two
/// ownership roles for every level and every modality — `pack_offsets` and
/// `pack_owner` — so a program that emitted an owner map under the role `pixels`
/// would be handing its caller a tensor it could only guess at.
///
/// This reaches only declarations that have a role to check, which is what makes
/// it enforceable rather than merely desirable. A preprocessing program declares
/// what each of its outputs *is*, so a program that wires some other plausible
/// int64 rank-1 vector into a level is caught here instead of at the first
/// split. A companion a component's graph produces has no such declaration — it
/// is an ONNX output port, and a port contract has nowhere to carry a content
/// role — so a level whose extent is `produced` is not asked for something its
/// half of the schema does not have.
///
/// Padding deliberately does not get the same treatment, and the difference is
/// not an inconsistency. A length vector already has established modality
/// spellings that programs legitimately emit, so no single role could be
/// demanded without breaking a program that predates the generic name. The two
/// ownership roles are new, with no legacy to accommodate, so requiring them
/// rejects nothing that exists. Both rules resolve the *reference* by name; this
/// one additionally constrains what the referenced declaration may claim to be.
fn validate_program_companion_roles(
    kind: &str,
    outputs: &[PreprocessingOutputView<'_>],
    errors: &mut Vec<String>,
) {
    let role_of: BTreeMap<&str, &str> = outputs
        .iter()
        .map(|output| (output.name, output.content))
        .collect();
    let mut require = |referrer: &str, companion: &str, allowed: &[&str], what: &str| {
        let Some(role) = role_of.get(companion) else {
            return;
        };
        if allowed.contains(role) {
            return;
        }
        errors.push(format!(
            "preprocessing.{kind} output '{referrer}' names '{companion}' as its {what}, but that \
             output carries content role '{role}'; a companion says what it is by its role, and \
             the roles that say this are {}",
            allowed
                .iter()
                .map(|role| format!("'{role}'"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    };
    for output in outputs {
        let Some(contract) = output.contract else {
            continue;
        };
        // A `padding` entry names its length value by name, not by role. The
        // audio vocabulary already spells a length `frame_lengths` and
        // `sample_lengths`, and a program pointing its padding entry at the one
        // it already emits is right rather than in need of a second output, so
        // nothing here dispatches on which spelling it chose.
        for level in contract.batch_layout.levels() {
            require(
                output.name,
                &level.offsets,
                &[crate::schema::PACK_OFFSETS_CONTENT],
                "ownership offsets",
            );
            require(
                output.name,
                &level.owner,
                &[crate::schema::PACK_OWNER_CONTENT],
                "ownership owner map",
            );
        }
    }
}

fn validate_preprocessing_workflow(metadata: &InferenceMetadata, errors: &mut Vec<String>) {
    let Some(workflow) = metadata
        .pipeline
        .as_ref()
        .map(|pipeline| &pipeline.workflow)
    else {
        return;
    };
    let preprocessing = metadata.preprocessing.as_ref();
    let programs = [
        (
            "image",
            "onnx-genai.image-preprocess",
            preprocessing
                .and_then(|spec| spec.image.as_ref())
                .map(PreprocessingOutputView::pixels),
        ),
        (
            "video",
            "onnx-genai.video-preprocess",
            preprocessing
                .and_then(|spec| spec.video.as_ref())
                .map(PreprocessingOutputView::pixels),
        ),
        (
            "audio",
            "onnx-genai.audio-preprocess",
            preprocessing
                .and_then(|spec| spec.audio.as_ref())
                .map(PreprocessingOutputView::audio),
        ),
    ];
    for (kind, abi, program_outputs) in programs {
        validate_preprocessing_program(workflow, kind, abi, program_outputs, errors);
    }
}

fn validate_preprocessing_program(
    workflow: &crate::schema::WorkflowSpec,
    kind: &str,
    abi: &str,
    program_outputs: Option<Vec<PreprocessingOutputView<'_>>>,
    errors: &mut Vec<String>,
) {
    const ABI_VERSION: &str = "1";

    let adapters = workflow
        .components
        .iter()
        .filter(|(_, component)| {
            matches!(
                &component.implementation,
                crate::schema::ComponentImplementation::Adapter { abi: component_abi, version, .. }
                    if component_abi == abi && version == ABI_VERSION
            )
        })
        .collect::<Vec<_>>();
    if program_outputs.is_none() && !adapters.is_empty() {
        errors.push(format!(
            "workflow adapter components using {abi}@{ABI_VERSION} require \
                 preprocessing.{kind} metadata"
        ));
        return;
    }
    let Some(program_outputs) = program_outputs else {
        return;
    };
    validate_program_companion_roles(kind, &program_outputs, errors);
    if adapters.len() != 1 {
        errors.push(format!(
            "preprocessing.{kind} requires exactly one workflow adapter component using \
                 {abi}@{ABI_VERSION}, found {}",
            adapters.len()
        ));
        return;
    }
    let (adapter_name, adapter) = adapters[0];
    match adapter.ports.inputs.get("encoded") {
        Some(contract) if contract.dtype == "uint8" && contract.rank == 1 => {}
        Some(contract) => errors.push(format!(
            "workflow {kind} preprocessing adapter '{adapter_name}' input 'encoded' must be uint8 \
                 rank 1, got {} rank {}",
            contract.dtype, contract.rank
        )),
        None => errors.push(format!(
            "workflow {kind} preprocessing adapter '{adapter_name}' must declare input 'encoded'"
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
            | WorkflowNode::Transfer { .. }
            | WorkflowNode::ExecutionIsland { .. } => {}
        }
    }
    let Ok(compiled) = crate::compile_workflow(workflow) else {
        return;
    };
    let mut invocations = Vec::new();
    collect_invocations(&compiled.graph, adapter_name, &mut invocations);
    if invocations.len() != 1 {
        errors.push(format!(
            "workflow {kind} preprocessing adapter '{adapter_name}' must be invoked exactly once, \
                 found {} invocations",
            invocations.len()
        ));
        return;
    }
    let invocation_outputs = invocations[0];
    for output in &program_outputs {
        if output.optional {
            errors.push(format!(
                "preprocessing.{kind} output '{}' cannot be optional in a workflow; every declared \
                 adapter SSA output must be materialized",
                output.name
            ));
        }
        let Some(contract) = output.contract else {
            errors.push(format!(
                "preprocessing.{kind} output '{}' must declare a TensorContract for workflow use",
                output.name
            ));
            continue;
        };
        if contract.dtype != output.dtype {
            errors.push(format!(
                "preprocessing.{kind} output '{}' dtype '{}' disagrees with its TensorContract '{}'",
                output.name, output.dtype, contract.dtype
            ));
        }
        let port = invocation_outputs
            .iter()
            .find_map(|(port, value)| (value == output.name).then_some(port));
        let Some(port) = port else {
            errors.push(format!(
                "preprocessing.{kind} output '{}' must be a declared SSA output of adapter \
                     invocation '{adapter_name}'",
                output.name
            ));
            continue;
        };
        match adapter.ports.outputs.get(port) {
            Some(port_contract) => require_compatible_tensor_contracts(
                contract,
                port_contract,
                &format!("preprocessing.{kind} output '{}'", output.name),
                errors,
            ),
            None => errors.push(format!(
                "workflow {kind} preprocessing adapter '{adapter_name}' has no output port '{port}'"
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
    // batch_layout and padding are part of the contract: a preprocessing output
    // that claims a different row correspondence, or a different notion of which
    // entries are real, than the port it feeds would let per-request rows drift
    // out of alignment with the rest of the workflow.
    if normalize(&source.dtype) != normalize(&target.dtype)
        || source.rank != target.rank
        || source.shape != target.shape
        || source.batch_layout != target.batch_layout
        || source.padding != target.padding
    {
        errors.push(format!(
            "{path} has a contract incompatible with its adapter output port"
        ));
    }
}

/// All structural problems found in a pipeline specification.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid pipeline spec: {errors:?}")]
pub struct PipelineValidationError {
    pub errors: Vec<String>,
}

/// Validate the pipeline DAG and component references.
///
/// `version` is the schema version the document declares, normalized. Almost
/// every rule here is version-independent, because almost every rule describes a
/// document that never worked. A rule that instead tightens what a *working*
/// document must say is scoped to the version that introduced it, so a package
/// on the shelf keeps loading: see [`validate_emit_axis`].
pub fn validate_pipeline_spec(
    spec: &PipelineSpec,
    version: crate::version::SchemaVersion,
) -> Result<(), PipelineValidationError> {
    let mut errors = Vec::new();
    validate_workflow(&spec.workflow, version, &mut errors);
    if errors.is_empty() {
        Ok(())
    } else {
        Err(PipelineValidationError { errors })
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_adapter_selection_input(
    workflow: &WorkflowSpec,
    name: &str,
    dtype: &str,
    rank: usize,
    second_dimension: Option<usize>,
    expected_role: Option<crate::schema::RuntimeInputRole>,
    field: &str,
    errors: &mut Vec<String>,
) {
    let expected_shape = |shape: &[crate::schema::TensorDimension]| match second_dimension {
        None => matches!(
            shape,
            [crate::schema::TensorDimension::Symbol(symbol)] if symbol == "batch"
        ),
        Some(extent) => matches!(
            shape,
            [
                crate::schema::TensorDimension::Symbol(symbol),
                crate::schema::TensorDimension::Fixed(actual)
            ] if symbol == "batch" && *actual == extent as i64
        ),
    };
    match workflow.inputs.get(name) {
        Some(input)
            if input.contract.dtype == dtype
                && input.contract.rank == rank
                && input.contract.shape.as_deref().is_some_and(expected_shape)
                && input.contract.batch_layout
                    == crate::schema::BatchLayout::RequestAligned { axis: 0 }
                && input.required
                && matches!(
                    &input.source,
                    crate::schema::WorkflowInputSource::Request
                        | crate::schema::WorkflowInputSource::Application { .. }
                )
                && expected_role.as_ref().is_none_or(|expected_role| {
                    matches!(
                        &input.role,
                        crate::schema::SemanticInputRole::Runtime { version, role }
                            if version == "1.0" && role == expected_role
                    )
                }) => {}
        Some(_) => errors.push(format!(
            "adapters.selection.{field} '{name}' must reference a required \
             request/application-sourced {dtype}{} workflow input with \
             batch_layout request_aligned on axis 0{}",
            if let Some(extent) = second_dimension {
                format!("[batch,{extent}]")
            } else {
                "[batch]".to_string()
            },
            expected_role
                .map(|role| format!(" and runtime role {role:?}@1.0"))
                .unwrap_or_default()
        )),
        None => errors.push(format!(
            "adapters.selection.{field} '{name}' references an undeclared workflow input"
        )),
    }
}

fn validate_adapter_service(
    service: &crate::schema::AdapterServiceContract,
    workflow: Option<&WorkflowSpec>,
    errors: &mut Vec<String>,
) {
    if service.application_capability.trim().is_empty() {
        errors.push("adapters.application_capability must not be empty".into());
    } else if service.application_capability != "onnx-genai.adapters@1" {
        errors.push("adapters.application_capability must be onnx-genai.adapters@1".into());
    }
    if service.cache.max_entries == 0 {
        errors.push("adapters.cache.max_entries must be greater than zero".into());
    }
    if service.selection.max_adapters == 0 {
        errors.push("adapters.selection.max_adapters must be greater than zero".into());
    }
    if service.artifacts.is_empty() {
        errors.push("adapters.artifacts must not be empty".into());
    }
    if let Some(workflow) = workflow {
        validate_adapter_selection_input(
            workflow,
            &service.selection.segments,
            "int64",
            2,
            Some(service.selection.max_adapters),
            Some(crate::schema::RuntimeInputRole::AdapterSegments),
            "segments",
            errors,
        );
        validate_adapter_selection_input(
            workflow,
            &service.selection.adapter_counts,
            "int64",
            1,
            None,
            Some(crate::schema::RuntimeInputRole::AdapterCounts),
            "adapter_counts",
            errors,
        );
        validate_adapter_selection_input(
            workflow,
            &service.selection.scales,
            "float32",
            2,
            Some(service.selection.max_adapters),
            Some(crate::schema::RuntimeInputRole::AdapterScales),
            "scales",
            errors,
        );
        if let Some(active) = &service.selection.active {
            validate_adapter_selection_input(
                workflow,
                active,
                "bool",
                1,
                None,
                Some(crate::schema::RuntimeInputRole::AdapterActive),
                "active",
                errors,
            );
        }
    } else {
        for (field, value) in [
            ("segments", service.selection.segments.as_str()),
            ("adapter_counts", service.selection.adapter_counts.as_str()),
            ("scales", service.selection.scales.as_str()),
        ] {
            if value.trim().is_empty() {
                errors.push(format!("adapters.selection.{field} must not be empty"));
            }
        }
    }
    let mut identities = BTreeSet::new();
    let mut indices = BTreeSet::new();
    let mut manifest_targets = BTreeMap::new();
    let mut resolved_targets = BTreeSet::new();
    for (index, target) in service.target_manifest.targets.iter().enumerate() {
        let path = format!("adapters.target_manifest.targets[{index}]");
        if target.id.trim().is_empty()
            || target.component.trim().is_empty()
            || target.initializer.trim().is_empty()
            || target.node_name.trim().is_empty()
            || target.output_name.trim().is_empty()
        {
            errors.push(format!(
                "{path} id, component, initializer, node_name, and output_name must not be empty"
            ));
        }
        if manifest_targets.insert(target.id.clone(), target).is_some() {
            errors.push(format!("{path}.id '{}' is duplicated", target.id));
        }
        if !resolved_targets.insert((
            target.component.clone(),
            target.initializer.clone(),
            target.output_slice.as_ref().map(|slice| slice.role.clone()),
            target.output_slice.as_ref().map(|slice| slice.offset),
            target.output_slice.as_ref().map(|slice| slice.width),
        )) {
            errors.push(format!(
                "{path} duplicates a resolved component/initializer/slice binding"
            ));
        }
        if target.input_features == 0 || target.output_features == 0 {
            errors.push(format!(
                "{path} input_features and output_features must be greater than zero"
            ));
        }
        if !matches!(
            target.activation_dtype.as_str(),
            "float16" | "fp16" | "float32" | "fp32" | "bfloat16" | "bf16"
        ) {
            errors.push(format!(
                "{path}.activation_dtype '{}' must be a floating-point tensor dtype",
                target.activation_dtype
            ));
        }
        if target.rank == Some(0) {
            errors.push(format!(
                "{path}.rank must be greater than zero when present"
            ));
        }
        if target
            .alpha
            .is_some_and(|alpha| !alpha.is_finite() || alpha <= 0.0)
        {
            errors.push(format!(
                "{path}.alpha must be finite and greater than zero when present"
            ));
        }
        if let Some(slice) = &target.output_slice
            && (slice.role.trim().is_empty()
                || slice.width == 0
                || slice.rank == Some(0)
                || slice
                    .alpha
                    .is_some_and(|alpha| !alpha.is_finite() || alpha <= 0.0)
                || slice
                    .offset
                    .checked_add(slice.width)
                    .is_none_or(|end| end > target.output_features))
        {
            errors.push(format!(
                "{path}.output_slice role must be non-empty and its range must be within output_features"
            ));
        }
        if let Some(native) = &target.graph_inputs
            && (native.a.trim().is_empty()
                || native.b.trim().is_empty()
                || native.a == native.b
                || native.scale.as_ref().is_some_and(|scale| {
                    scale.trim().is_empty() || scale == &native.a || scale == &native.b
                }))
        {
            errors.push(format!(
                "{path}.graph_inputs must contain distinct non-empty a/b/optional-scale input names"
            ));
        }
        match workflow {
            Some(workflow) if !workflow.components.contains_key(&target.component) => {
                errors.push(format!(
                    "{path}.component '{}' is undeclared",
                    target.component
                ));
            }
            None if target.component != "model" => errors.push(format!(
                "{path}.component must be 'model' for bare-model metadata"
            )),
            _ => {}
        }
    }
    if service.target_manifest.targets.is_empty() {
        errors.push("adapters.target_manifest.targets must not be empty".into());
    }
    for (name, artifact) in &service.artifacts {
        let path = format!("adapters.artifacts.{name}");
        if artifact.identity.trim().is_empty() || artifact.version.trim().is_empty() {
            errors.push(format!("{path} identity and version must not be empty"));
        }
        if let Some(provenance) = &artifact.provenance
            && (provenance.producer.trim().is_empty()
                || provenance
                    .source
                    .as_ref()
                    .is_some_and(|value| value.trim().is_empty())
                || provenance
                    .revision
                    .as_ref()
                    .is_some_and(|value| value.trim().is_empty()))
        {
            errors.push(format!(
                "{path}.provenance producer and present source/revision values must not be empty"
            ));
        }
        if !indices.insert(artifact.index) {
            errors.push(format!(
                "{path}.index {} duplicates another adapter wire ID",
                artifact.index
            ));
        }
        if !identities.insert((artifact.identity.clone(), artifact.version.clone())) {
            errors.push(format!(
                "{path} duplicates adapter identity {}@{}",
                artifact.identity, artifact.version
            ));
        }
        if artifact.rank == 0 {
            errors.push(format!("{path}.rank must be greater than zero"));
        }
        if !artifact.alpha.is_finite() || artifact.alpha <= 0.0 {
            errors.push(format!("{path}.alpha must be finite and greater than zero"));
        }
        if !matches!(
            artifact.dtype.as_str(),
            "float16" | "fp16" | "float32" | "fp32" | "bfloat16" | "bf16"
        ) {
            errors.push(format!(
                "{path}.dtype '{}' must be a floating-point adapter dtype",
                artifact.dtype
            ));
        }
        if artifact.weights.is_empty() {
            errors.push(format!(
                "{path}.weights must declare at least one external artifact"
            ));
        }
        let mut weight_formats = BTreeSet::new();
        for (index, weight) in artifact.weights.iter().enumerate() {
            if !weight_formats.insert(weight.format.clone()) {
                errors.push(format!(
                    "{path}.weights contains duplicate format {:?}",
                    weight.format
                ));
            }
            if weight.location.trim().is_empty()
                || std::path::Path::new(&weight.location).is_absolute()
                || !weight.location.starts_with(&format!("adapters/{name}/"))
                || weight
                    .location
                    .split(['/', '\\'])
                    .any(|segment| segment == "..")
            {
                errors.push(format!(
                    "{path}.weights[{index}].location must be under package path adapters/{name}/"
                ));
            }
            if weight.loader_capability.trim().is_empty() {
                errors.push(format!(
                    "{path}.weights[{index}].loader_capability must not be empty"
                ));
            }
            let expected_loader = match weight.format {
                crate::schema::AdapterWeightFormat::Json => "onnx-genai.adapters.json@1",
                crate::schema::AdapterWeightFormat::OrtGenai => "onnxruntime.lora-adapter@1",
                crate::schema::AdapterWeightFormat::HfPeft => "onnx-genai.adapters.hf-peft@1",
                crate::schema::AdapterWeightFormat::Safetensors => {
                    "onnx-genai.adapters.safetensors@1"
                }
            };
            if weight.loader_capability != expected_loader {
                errors.push(format!(
                    "{path}.weights[{index}].loader_capability must be {expected_loader} for format {:?}",
                    weight.format
                ));
            }
            match (&weight.format, &weight.scale_encoding) {
                (
                    crate::schema::AdapterWeightFormat::HfPeft,
                    crate::schema::AdapterScaleEncoding::AlphaOverRank,
                )
                | (
                    crate::schema::AdapterWeightFormat::OrtGenai,
                    crate::schema::AdapterScaleEncoding::Baked,
                )
                | (crate::schema::AdapterWeightFormat::Json, _)
                | (crate::schema::AdapterWeightFormat::Safetensors, _) => {}
                (crate::schema::AdapterWeightFormat::HfPeft, _) => errors.push(format!(
                    "{path}.weights[{index}].scale_encoding must be alpha_over_rank for hf_peft"
                )),
                (crate::schema::AdapterWeightFormat::OrtGenai, _) => errors.push(format!(
                    "{path}.weights[{index}].scale_encoding must be baked for ort_genai"
                )),
            }
            let peft = weight.format == crate::schema::AdapterWeightFormat::HfPeft;
            if peft != weight.config_location.is_some() {
                errors.push(format!(
                    "{path}.weights[{index}] hf_peft requires config_location; other formats forbid it"
                ));
            }
            if let Some(config_location) = &weight.config_location
                && (std::path::Path::new(config_location).is_absolute()
                    || !config_location.starts_with(&format!("adapters/{name}/"))
                    || config_location
                        .split(['/', '\\'])
                        .any(|segment| segment == ".."))
            {
                errors.push(format!(
                    "{path}.weights[{index}].config_location must be under package path adapters/{name}/"
                ));
            }
        }
        if artifact.bindings.is_empty() {
            errors.push(format!("{path}.bindings must not be empty"));
        }
        let mut local_targets = BTreeSet::new();
        let mut local_weight_keys = BTreeSet::new();
        let has_ort_bundle = artifact
            .weights
            .iter()
            .any(|weight| weight.format == crate::schema::AdapterWeightFormat::OrtGenai);
        for (index, binding) in artifact.bindings.iter().enumerate() {
            let target_path = format!("{path}.bindings[{index}]");
            if binding.target.trim().is_empty() || binding.weight_key.trim().is_empty() {
                errors.push(format!(
                    "{target_path} target and weight_key must not be empty"
                ));
            }
            if !local_weight_keys.insert(binding.weight_key.clone()) {
                errors.push(format!(
                    "{path} declares duplicate weight_key '{}'",
                    binding.weight_key
                ));
            }
            let Some(target) = manifest_targets.get(&binding.target) else {
                errors.push(format!(
                    "{target_path}.target '{}' is absent from adapters.target_manifest",
                    binding.target
                ));
                continue;
            };
            if has_ort_bundle && target.graph_inputs.is_none() {
                errors.push(format!(
                    "{target_path}.target '{}' requires graph_inputs for an ort_genai weight artifact",
                    binding.target
                ));
            }
            if binding.rank == Some(0) {
                errors.push(format!(
                    "{target_path}.rank must be greater than zero when present"
                ));
            }
            if binding
                .alpha
                .is_some_and(|alpha| !alpha.is_finite() || alpha <= 0.0)
            {
                errors.push(format!(
                    "{target_path}.alpha must be finite and greater than zero when present"
                ));
            }
            let effective_rank = binding.rank.unwrap_or(artifact.rank);
            let effective_alpha = binding.alpha.unwrap_or(artifact.alpha);
            if target.rank.is_some_and(|rank| rank != effective_rank) {
                errors.push(format!(
                    "{target_path} effective rank {effective_rank} violates target policy {:?}",
                    target.rank
                ));
            }
            if target.alpha.is_some_and(|alpha| alpha != effective_alpha) {
                errors.push(format!(
                    "{target_path} effective alpha {effective_alpha} violates target policy {:?}",
                    target.alpha
                ));
            }
            if let Some(slice) = &target.output_slice {
                if slice.rank.is_some_and(|rank| rank != effective_rank) {
                    errors.push(format!(
                        "{target_path} effective rank {effective_rank} violates output-slice policy {:?}",
                        slice.rank
                    ));
                }
                if slice.alpha.is_some_and(|alpha| alpha != effective_alpha) {
                    errors.push(format!(
                        "{target_path} effective alpha {effective_alpha} violates output-slice policy {:?}",
                        slice.alpha
                    ));
                }
            }
            if !local_targets.insert(binding.target.clone()) {
                errors.push(format!(
                    "{path} declares duplicate target '{}'",
                    binding.target
                ));
            }
        }
    }
    if indices
        .iter()
        .copied()
        .enumerate()
        .any(|(expected, actual)| expected != actual)
    {
        errors.push(format!(
            "adapters artifact indices must be contiguous from zero; found {indices:?}"
        ));
    }
    let Some(workflow) = workflow else {
        return;
    };
    for (name, component) in &workflow.components {
        let Some(contract) = component
            .contract
            .as_ref()
            .filter(|contract| contract.id == "onnx-genai.parameter-overlay")
        else {
            continue;
        };
        let string_parameter = |key: &str| match contract.parameters.get(key) {
            Some(crate::schema::ScalarValue::String(value)) if !value.trim().is_empty() => {
                Some(value.as_str())
            }
            _ => None,
        };
        let target_component = string_parameter("component");
        let target_parameter = string_parameter("parameter");
        if target_component.is_none() {
            errors.push(format!(
                "workflow parameter overlay component '{name}' must declare non-empty string parameter 'component'"
            ));
        }
        if target_parameter.is_none() {
            errors.push(format!(
                "workflow parameter overlay component '{name}' must declare non-empty string parameter 'parameter'"
            ));
        }
        if let (Some(target_component), Some(target_parameter)) =
            (target_component, target_parameter)
            && !service.target_manifest.targets.iter().any(|target| {
                target.component == target_component && target.initializer == target_parameter
            })
        {
            errors.push(format!(
                "workflow parameter overlay component '{name}' targets undeclared adapter parameter '{target_component}.{target_parameter}'"
            ));
        }
    }
}

fn validate_workflow(
    workflow: &WorkflowSpec,
    version: crate::version::SchemaVersion,
    errors: &mut Vec<String>,
) {
    let compiled = match crate::compile_workflow(workflow) {
        Ok(compiled) => compiled,
        Err(error) => {
            errors.push(format!("pipeline.workflow lowering failed: {error}"));
            return;
        }
    };
    /// Check that a workflow tensor names a dtype this format can talk about.
    ///
    /// This is a vocabulary check, not a capability check. Metadata is portable:
    /// it cannot know which execution provider will load the package, so it must
    /// not refuse a dtype merely because some runtime lacks a kernel for it. FP8
    /// caches are the motivating case — a package may legitimately store its KV
    /// state as `float8_e4m3fn`, and whether a given EP can compute on that is
    /// answered at binding time by the runtime, with a capability error that
    /// names the EP. Rejecting it here would report an execution-provider gap as
    /// a malformed document.
    fn validate_runtime_dtype(
        path: &str,
        contract: &crate::schema::TensorContract,
        errors: &mut Vec<String>,
    ) {
        if let crate::schema::BatchLayout::RequestExpanded { factor: 0, .. } = contract.batch_layout
        {
            errors.push(format!(
                "{path} declares request_expanded.factor 0; the expansion factor must be at least 1"
            ));
        }
        if contract
            .shape
            .as_ref()
            .is_some_and(|shape| shape.len() != contract.rank)
        {
            errors.push(format!(
                "{path} declares rank {} but has {} shape dimensions",
                contract.rank,
                contract.shape.as_ref().map_or(0, Vec::len)
            ));
        }
        if !matches!(
            contract.dtype.as_str(),
            "float16"
                | "fp16"
                | "float32"
                | "fp32"
                | "bfloat16"
                | "bf16"
                | "float8_e4m3fn"
                | "fp8_e4m3fn"
                | "float8_e4m3"
                | "fp8_e4m3"
                | "float8_e5m2"
                | "fp8_e5m2"
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
                "{path} uses dtype '{}', which is not a tensor dtype this format can name",
                contract.dtype
            ));
        }
    }

    let mut initial_value_names = workflow.inputs.keys().cloned().collect::<BTreeSet<_>>();
    for (name, input) in &workflow.inputs {
        validate_runtime_dtype(&format!("workflow input '{name}'"), &input.contract, errors);
        if matches!(input.source, crate::schema::WorkflowInputSource::Request)
            && !matches!(input.role, crate::schema::SemanticInputRole::Runtime { .. })
        {
            errors.push(format!(
                "workflow request input '{name}' must declare one versioned runtime role"
            ));
        }
        if !input.required && input.default.is_none() && input.present_as.is_none() {
            errors.push(format!(
                "workflow input '{name}' cannot be optional without a literal default or \
                 present_as predicate"
            ));
        }
        if let Some(present_as) = &input.present_as {
            if input.required {
                errors.push(format!(
                    "workflow input '{name}' uses present_as but is required; presence predicates \
                     are only valid for optional inputs"
                ));
            }
            if input.default.is_some() {
                errors.push(format!(
                    "workflow input '{name}' cannot combine present_as with a literal default"
                ));
            }
            if !matches!(
                input.source,
                crate::schema::WorkflowInputSource::Request
                    | crate::schema::WorkflowInputSource::Application { .. }
            ) {
                errors.push(format!(
                    "workflow input '{name}' can use present_as only with request or application \
                     sources"
                ));
            }
            if matches!(input.source, crate::schema::WorkflowInputSource::Request)
                && !matches!(
                    input.role,
                    crate::schema::SemanticInputRole::Runtime {
                        role: crate::schema::RuntimeInputRole::Media
                            | crate::schema::RuntimeInputRole::Constraint
                            | crate::schema::RuntimeInputRole::SessionId,
                        ..
                    }
                )
            {
                errors.push(format!(
                    "workflow request input '{name}' can use present_as only with media, \
                     constraint, or session_id roles whose absence is observable"
                ));
            }
            if present_as.trim().is_empty() {
                errors.push(format!(
                    "workflow input '{name}' has an empty present_as value"
                ));
            } else if !initial_value_names.insert(present_as.clone()) {
                errors.push(format!(
                    "workflow input '{name}' present_as value '{present_as}' collides with another \
                     initial SSA value"
                ));
            }
        }
    }
    for (name, input) in &workflow.inputs {
        if let Some(present_as) = &input.present_as {
            validate_optional_input_guards(
                &compiled.graph,
                name,
                present_as,
                false,
                "pipeline.workflow.steps",
                errors,
            );
        }
    }
    for (name, output) in &workflow.outputs {
        validate_runtime_dtype(
            &format!("workflow output '{name}'"),
            &output.contract,
            errors,
        );
        if let Some(media) = &output.media {
            if output.role != crate::schema::WorkflowOutputRole::Audio {
                errors.push(format!(
                    "workflow output '{name}' declares a media contract but its role is not audio"
                ));
            }
            if media.sample_rate_hz == Some(0) {
                errors.push(format!(
                    "workflow output '{name}' media.sample_rate_hz must be greater than zero"
                ));
            }
            if media.source_sample_rate_hz == Some(0) {
                errors.push(format!(
                    "workflow output '{name}' media.source_sample_rate_hz must be greater than zero"
                ));
            }
            if media.channels == Some(0) {
                errors.push(format!(
                    "workflow output '{name}' media.channels must be greater than zero"
                ));
            }
            if media.sample_rate_hz.is_none() {
                errors.push(format!(
                    "workflow output '{name}' audio media contract must declare sample_rate_hz"
                ));
            }
            if media.channels.is_none() {
                errors.push(format!(
                    "workflow output '{name}' audio media contract must declare channels"
                ));
            }
            if media.container == crate::schema::MediaContainer::Wav {
                match output.stage {
                    crate::schema::OutputStage::PostAdapter if output.contract.dtype != "uint8" => {
                        errors.push(format!(
                            "workflow output '{name}' is a post-adapter WAV but its contract dtype is not uint8"
                        ));
                    }
                    crate::schema::OutputStage::PreAdapter
                        if !matches!(
                            output.contract.dtype.as_str(),
                            "float32" | "fp32" | "float16" | "fp16" | "bfloat16" | "bf16"
                        ) =>
                    {
                        errors.push(format!(
                            "workflow output '{name}' is a pre-adapter WAV source but its contract dtype is not floating point"
                        ));
                    }
                    _ => {}
                }
            }
        }
        match (&output.role, output.value_range) {
            (crate::schema::WorkflowOutputRole::Image, None) => errors.push(format!(
                "workflow image output '{name}' must declare value_range"
            )),
            (crate::schema::WorkflowOutputRole::Image, Some(_)) => {}
            (_, Some(_)) => errors.push(format!(
                "workflow non-image output '{name}' cannot declare image value_range"
            )),
            (_, None) => {}
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
            } => match workflow.manifest.adapter_abis.get(abi) {
                Some(pinned) if pinned == version => {}
                _ => errors.push(format!(
                    "workflow component '{name}' requires adapter ABI {abi}@{version}, \
                         but the manifest does not pin that exact version"
                )),
            },
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
        if let Some(contract) = &component.contract {
            if contract.id.trim().is_empty() || contract.version.trim().is_empty() {
                errors.push(format!(
                    "workflow component '{name}' contract id and version must not be empty"
                ));
            }
            if !(component.ports.inputs.is_empty() && component.ports.outputs.is_empty()) {
                for (role, port) in &contract.bindings {
                    if role.trim().is_empty()
                        || (!component.ports.inputs.contains_key(port)
                            && !component.ports.outputs.contains_key(port))
                    {
                        errors.push(format!(
                            "workflow component '{name}' contract binding '{role}' references \
                             unknown port '{port}'"
                        ));
                    }
                }
            }
            if let crate::schema::ComponentImplementation::Adapter { abi, version, .. } =
                &component.implementation
            {
                if contract.id != *abi || contract.version != *version {
                    errors.push(format!(
                        "workflow adapter component '{name}' contract {}@{} must match its ABI \
                             {abi}@{version}",
                        contract.id, contract.version
                    ));
                }
                let action = match contract.parameters.get("action") {
                    Some(crate::schema::ScalarValue::String(action)) => Some(action.as_str()),
                    _ => None,
                };
                let known_contract = CONTRACT_OBLIGATIONS
                    .iter()
                    .any(|entry| entry.id == contract.id && entry.version == contract.version);
                let obligation = CONTRACT_OBLIGATIONS.iter().find(|entry| {
                    entry.id == contract.id
                        && entry.version == contract.version
                        && action == Some(entry.action)
                });
                if known_contract && obligation.is_none() {
                    errors.push(format!(
                        "workflow adapter component '{name}' has unsupported action {:?} for \
                             contract {}@{}",
                        action, contract.id, contract.version
                    ));
                }
                for role in obligation
                    .into_iter()
                    .flat_map(|entry| entry.inputs.iter().chain(entry.outputs))
                {
                    if !contract.bindings.contains_key(*role) {
                        errors.push(format!(
                            "workflow adapter component '{name}' contract {}@{} is missing \
                             required binding '{role}'",
                            contract.id, contract.version
                        ));
                    }
                }
                for (direction, roles, ports) in obligation.into_iter().flat_map(|entry| {
                    [
                        ("input", entry.inputs, &component.ports.inputs),
                        ("output", entry.outputs, &component.ports.outputs),
                    ]
                }) {
                    for role in roles {
                        if contract
                            .bindings
                            .get(*role)
                            .is_some_and(|port| !ports.contains_key(port))
                        {
                            errors.push(format!(
                                "workflow adapter component '{name}' contract role '{role}' must \
                                 bind an {direction} port"
                            ));
                        }
                    }
                }
            }
        }
        if component.application_overridable && component.contract.is_none() {
            errors.push(format!(
                "workflow component '{name}' is application-overridable but has no versioned contract"
            ));
        }
        if component.application_overridable
            && !matches!(
                component.implementation,
                crate::schema::ComponentImplementation::Onnx { .. }
            )
        {
            errors.push(format!(
                "workflow component '{name}' is application-overridable but is not an ONNX component"
            ));
        }
    }
    for (name, state) in &workflow.state {
        validate_runtime_dtype(&format!("workflow state '{name}'"), &state.contract, errors);
        if state.scope == crate::schema::WorkflowStateScope::Invocation && state.session.is_some() {
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

        let dynamic_axis = match &state.recurrence {
            crate::schema::ShapeRecurrence::Growing { axis, .. }
            | crate::schema::ShapeRecurrence::Bounded { axis, .. } => Some(*axis),
            crate::schema::ShapeRecurrence::Invariant => None,
        };
        if let Some(axis) = dynamic_axis
            && axis >= state.contract.rank
        {
            errors.push(format!(
                "workflow state '{name}' varies on axis {axis}, outside rank {}",
                state.contract.rank
            ));
        }
        if let Some(group_name) = &state.service_group {
            let group = workflow
                .serving
                .as_ref()
                .and_then(|serving| serving.state_service.groups.get(group_name));
            let Some(group) = group else {
                errors.push(format!(
                    "workflow state '{name}' binds unknown state service group '{group_name}'"
                ));
                continue;
            };
            if let Some(sequence_axis) = group.sequence_axis {
                if sequence_axis >= state.contract.rank {
                    errors.push(format!(
                        "state service group '{group_name}' sequence_axis {sequence_axis} is outside state '{name}' rank {}",
                        state.contract.rank
                    ));
                }
                if dynamic_axis.is_some_and(|axis| axis != sequence_axis) {
                    errors.push(format!(
                        "workflow state '{name}' recurrence axis {dynamic_axis:?} disagrees with \
                         state service group '{group_name}' sequence_axis {sequence_axis}"
                    ));
                }
            } else if dynamic_axis.is_some() {
                errors.push(format!(
                    "workflow state '{name}' varies along axis {dynamic_axis:?}, but state \
                     service group '{group_name}' declares no sequence_axis"
                ));
            }
            // Row-scoped model state must remain permutable: without a declared
            // request axis a runtime cannot compact the batch correctly.
            if state.contract.batch_layout.request_axis().is_none() {
                errors.push(format!(
                    "workflow state '{name}' binds state service group '{group_name}' but its \
                     contract does not declare a request_aligned batch_layout; compaction would \
                     be underivable"
                ));
            }
            if let crate::schema::ShapeRecurrence::Growing { increment, .. } = &state.recurrence
                && let Some(serving) = &workflow.serving
                && serving.accepted_len.as_deref() != Some(increment)
            {
                errors.push(format!(
                    "state service state '{name}' grows by '{increment}', but serving.accepted_len \
                     does not bind that per-row value"
                ));
            }
        }
    }

    if let Some(state_service) = workflow
        .serving
        .as_ref()
        .map(|serving| &serving.state_service)
    {
        for (group_name, group) in &state_service.groups {
            if group.layout.trim().is_empty() {
                errors.push(format!(
                    "state service group '{group_name}' layout must not be empty"
                ));
            }
            if let Some(logical_lengths) = &group.logical_lengths {
                match workflow.state.get(logical_lengths) {
                    Some(lengths) => {
                        validate_integer_control_contract(
                            &lengths.contract,
                            &format!(
                                "state service group '{group_name}' logical_lengths state \
                                 '{logical_lengths}'"
                            ),
                            errors,
                        );
                        if lengths.contract.rank != 1 {
                            errors.push(format!(
                                "state service group '{group_name}' logical_lengths state \
                                 '{logical_lengths}' must be rank one with one value per row"
                            ));
                        }
                        if lengths.class != crate::schema::WorkflowStateClass::Semantic {
                            errors.push(format!(
                                "state service group '{group_name}' logical_lengths state \
                                 '{logical_lengths}' must be semantic for checkpoint/replay"
                            ));
                        }
                    }
                    None => errors.push(format!(
                        "state service group '{group_name}' references unknown logical_lengths \
                         state '{logical_lengths}'"
                    )),
                }
            }
            if let Some(total_length) = &group.total_length {
                require_workflow_value(
                    total_length,
                    &workflow.state.keys().cloned().collect(),
                    &format!("state service group '{group_name}' total_length"),
                    errors,
                );
            }
            validate_state_update(group_name, group, workflow, errors);
            validate_state_port_layers(group_name, group, errors);
            validate_attention_component_declares_sequence_role(
                group_name, group, workflow, errors,
            );
            for cascade in &group.capabilities.cascade {
                if !state_service.groups.contains_key(cascade) {
                    errors.push(format!(
                        "state service group '{group_name}' cascades to unknown group '{cascade}'"
                    ));
                }
            }
            if group.capabilities.rollback_positions == Some(0) {
                errors.push(format!(
                    "state service group '{group_name}' rollback_positions must be greater than \
                     zero when present; omit it to declare that rollback is impossible"
                ));
            }
            for (component_name, cells) in &group.ports {
                let Some(component) = workflow.components.get(component_name) else {
                    errors.push(format!(
                        "state service group '{group_name}' binds unknown component \
                         '{component_name}'"
                    ));
                    continue;
                };
                let inferred_ports = component.ports.inputs.is_empty()
                    && component.ports.outputs.is_empty()
                    && matches!(
                        component.implementation,
                        crate::schema::ComponentImplementation::Onnx { .. }
                    );
                for (cell_name, alias) in cells {
                    match workflow.state.get(cell_name) {
                        Some(state)
                            if state.service_group.as_deref() == Some(group_name.as_str()) => {}
                        Some(_) => errors.push(format!(
                            "state service group '{group_name}' port alias references state \
                             '{cell_name}' bound to another service group"
                        )),
                        None => errors.push(format!(
                            "state service group '{group_name}' port alias references unknown \
                             state '{cell_name}'"
                        )),
                    }
                    if !inferred_ports && !component.ports.inputs.contains_key(&alias.input) {
                        errors.push(format!(
                            "state service group '{group_name}' component '{component_name}' \
                             input alias '{}' is not a declared port",
                            alias.input
                        ));
                    }
                    match &alias.output {
                        Some(output) => {
                            if !inferred_ports && !component.ports.outputs.contains_key(output) {
                                errors.push(format!(
                                    "state service group '{group_name}' component \
                                     '{component_name}' output alias '{output}' is not a declared \
                                     port"
                                ));
                            }
                        }
                        None => {
                            // Only a read-only borrow may omit its output: a
                            // pure reader consumes a frozen buffer and advances
                            // nothing, so it exposes no present port. A
                            // read-write transition with no output could never
                            // be written back.
                            if alias.access != crate::schema::StatePortAccess::ReadOnly {
                                errors.push(format!(
                                    "state service group '{group_name}' component \
                                     '{component_name}' read-write alias for input '{}' declares \
                                     no output; only a read_only borrow may omit one",
                                    alias.input
                                ));
                            }
                        }
                    }
                }
            }
        }
    }

    let mut values = workflow.inputs.keys().cloned().collect::<BTreeSet<_>>();
    let mut value_contracts = workflow
        .inputs
        .iter()
        .map(|(name, input)| (name.clone(), input.contract.clone()))
        .collect::<BTreeMap<_, _>>();
    for input in workflow.inputs.values() {
        if let Some(present_as) = &input.present_as {
            values.insert(present_as.clone());
            value_contracts.insert(
                present_as.clone(),
                crate::schema::TensorContract {
                    dtype: "bool".to_string(),
                    rank: 0,
                    shape: Some(Vec::new()),
                    optional: false,
                    batch_layout: crate::schema::BatchLayout::Shared,
                    padding: Vec::new(),
                },
            );
        }
    }
    let mut effects = compiled.initial_effects.clone();
    let mut effect_tokens = effects.values().cloned().collect::<BTreeSet<_>>();
    validate_workflow_node(
        &compiled.graph,
        workflow,
        version,
        &mut values,
        &mut value_contracts,
        &mut effects,
        &mut effect_tokens,
        "pipeline.workflow.steps",
        errors,
    );
    validate_emit_batch_layout_consistency(
        &compiled.graph,
        &value_contracts,
        &mut BTreeMap::new(),
        errors,
    );
    validate_compaction_derivability(
        &compiled.graph,
        workflow,
        &value_contracts,
        "pipeline.workflow.steps",
        errors,
    );
    validate_effect_declarations(workflow, errors);
    validate_row_scoped_components(workflow, errors);
    validate_batch_layout_references(workflow, errors);
    validate_shared_companions(workflow, errors);
    validate_batch_capacity(workflow, errors);
    validate_state_lifetimes(workflow, errors);
    validate_session_continuity(workflow, errors);
    if let Some(serving) = &workflow.serving {
        if serving.state_service.groups.is_empty() {
            errors.push(
                "pipeline.workflow.serving.state_service.groups must declare at least one bound \
                 state group"
                    .to_string(),
            );
        }
        if !serving.state_service.groups.is_empty() && serving.accepted_len.is_none() {
            errors.push(
                "pipeline.workflow.serving.accepted_len is required when state service groups are \
                 declared"
                    .to_string(),
            );
        }
        for (role, value) in [
            ("active", Some(&serving.active)),
            ("done", Some(&serving.done)),
            ("accepted_len", serving.accepted_len.as_ref()),
        ] {
            let Some(value) = value else {
                continue;
            };
            require_workflow_value(
                value,
                &values,
                &format!("pipeline.workflow.serving.{role}"),
                errors,
            );
            if let Some(contract) = value_contracts.get(value) {
                if matches!(role, "active" | "done") {
                    validate_bool_control_contract(
                        contract,
                        &format!("pipeline.workflow.serving.{role}"),
                        errors,
                    );
                } else {
                    validate_integer_control_contract(
                        contract,
                        &format!("pipeline.workflow.serving.{role}"),
                        errors,
                    );
                }
                // Serving control values steer one request each. Without a
                // request-aligned layout a runtime cannot permute them when it
                // compacts the batch.
                if contract.rank > 0 && contract.batch_layout.request_axis() != Some(0) {
                    errors.push(format!(
                        "pipeline.workflow.serving.{role} '{value}' must declare a \
                         request_aligned batch_layout on axis 0"
                    ));
                }
            }
        }
    }

    let used = workflow_required_capabilities(workflow, Some(&compiled.graph));
    for capability in used.difference(&workflow.manifest.capabilities) {
        errors.push(format!(
            "pipeline.workflow.manifest.capabilities is missing used capability '{capability}'"
        ));
    }
}

fn workflow_state_results(node: &WorkflowNode) -> BTreeMap<String, String> {
    match node {
        WorkflowNode::Sequence { nodes } => {
            let mut results = BTreeMap::new();
            for node in nodes {
                results.extend(workflow_state_results(node));
            }
            results
        }
        WorkflowNode::Loop { setup, carried, .. } => {
            let mut results = workflow_state_results(setup);
            for carry in carried {
                results.insert(carry.cell.clone(), carry.next.clone());
            }
            results
        }
        WorkflowNode::Branch {
            cases,
            default,
            outputs,
            ..
        } => {
            let case_results = cases
                .iter()
                .map(|(case, node)| (case, workflow_state_results(node)))
                .collect::<BTreeMap<_, _>>();
            let default_results = default.as_deref().map(workflow_state_results);
            let mut results: BTreeMap<String, String> = BTreeMap::new();
            for (output, phi) in outputs {
                let mut cell: Option<String> = None;
                for (case, source) in &phi.cases {
                    let Some((candidate, _)) = case_results
                        .get(case)
                        .and_then(|results| results.iter().find(|(_, value)| *value == source))
                    else {
                        continue;
                    };
                    match &cell {
                        Some(expected) if expected != candidate => {
                            cell = None;
                            break;
                        }
                        None => cell = Some((*candidate).clone()),
                        _ => {}
                    }
                }
                if let (Some(source), Some(default_results)) = (&phi.default, &default_results)
                    && let Some((candidate, _)) =
                        default_results.iter().find(|(_, value)| *value == source)
                {
                    match &cell {
                        Some(expected) if expected != candidate => {
                            cell = None;
                        }
                        None => cell = Some(candidate.clone()),
                        _ => {}
                    }
                }
                if let Some(cell) = cell {
                    results.insert(cell, output.clone());
                }
            }
            results
        }
        _ => BTreeMap::new(),
    }
}

/// Every effect domain a component declares must have declared semantics.
///
/// Retry class and speculation safety are independent: a `transactional` effect
/// is still unsafe to speculate unless it also declares clonable or rewindable
/// speculation safety.
fn validate_effect_declarations(workflow: &WorkflowSpec, errors: &mut Vec<String>) {
    for (name, component) in &workflow.components {
        for domain in &component.effects {
            if domain.trim().is_empty() {
                errors.push(format!(
                    "workflow component '{name}' declares an empty effect domain"
                ));
            } else if !workflow.effects.contains_key(domain) {
                errors.push(format!(
                    "workflow component '{name}' declares effect domain '{domain}' that \
                     pipeline.workflow.effects does not describe; every effect must declare its \
                     retry class and speculation safety"
                ));
            }
        }
    }
    for (domain, contract) in &workflow.effects {
        if domain.trim().is_empty() {
            errors.push("pipeline.workflow.effects contains an empty domain name".to_string());
        }
        if matches!(
            contract.speculation_safety,
            crate::schema::SpeculationSafety::Rewindable { max_depth: 0 }
        ) {
            errors.push(format!(
                "pipeline.workflow.effects.{domain}.speculation_safety rewindable max_depth must \
                 be greater than zero; use kind 'none' to forbid speculation"
            ));
        }
    }
}

/// Row-scoped components must expose a derivable row axis on every row-scoped
/// port. The mandatory `compact(selection)`/`release(row)` ABI operates on that
/// axis, so an out-of-range declaration is unexecutable.
fn validate_row_scoped_components(workflow: &WorkflowSpec, errors: &mut Vec<String>) {
    for (name, component) in &workflow.components {
        let request_axes = component
            .ports
            .inputs
            .values()
            .chain(component.ports.outputs.values())
            .filter_map(|contract| contract.batch_layout.request_axis())
            .collect::<BTreeSet<_>>();
        let Some(row_scope) = &component.row_scope else {
            // A component that carries per-request state across invocations —
            // which is exactly what a declared effect domain or a declared
            // cache-affecting state means — must say which axis its rows lie on.
            // Row identities are never serialized, so without this the runtime
            // has nothing to drive compact/release with when the batch changes.
            if !request_axes.is_empty()
                && (!component.effects.is_empty() || !component.cache_affects_state.is_empty())
            {
                errors.push(format!(
                    "workflow component '{name}' holds per-request state and has request-aligned \
                     ports but declares no row_scope; the runtime cannot compact or release its \
                     rows without a declared row axis"
                ));
            }
            continue;
        };
        for (direction, ports) in [
            ("input", &component.ports.inputs),
            ("output", &component.ports.outputs),
        ] {
            for (port, contract) in ports {
                if !contract.batch_layout.is_row_scoped() {
                    continue;
                }
                if row_scope.axis >= contract.rank {
                    errors.push(format!(
                        "workflow component '{name}' declares row_scope axis {} but {direction} \
                         port '{port}' has rank {}",
                        row_scope.axis, contract.rank
                    ));
                }
                if contract
                    .batch_layout
                    .request_axis()
                    .is_some_and(|axis| axis != row_scope.axis)
                {
                    errors.push(format!(
                        "workflow component '{name}' {direction} port '{port}' is request-aligned \
                         on a different axis than the declared row_scope axis {}",
                        row_scope.axis
                    ));
                }
            }
        }
    }
}

/// Values a packed layout or a pad mask may name from one declaration site.
///
/// A workflow-level contract may only name workflow values. A component port
/// contract may additionally name a sibling port of the same component, because
/// the component's own ABI is what pairs a padded tensor with its lengths or a
/// packed tensor with its offsets; which SSA value reaches that port is the
/// invocation's business, not the port's.
#[derive(Clone, Copy)]
struct LayoutReferenceScope<'a> {
    declared: &'a BTreeSet<String>,
    contracts: &'a BTreeMap<String, crate::schema::TensorContract>,
    ports: Option<&'a crate::schema::ComponentPorts>,
}

impl LayoutReferenceScope<'_> {
    fn is_declared(&self, value: &str) -> bool {
        self.declared.contains(value)
            || self.ports.is_some_and(|ports| {
                ports.inputs.contains_key(value) || ports.outputs.contains_key(value)
            })
    }

    /// The contract of a referenced value, when the document states one.
    ///
    /// A value produced by control flow or by a component with inferred ports
    /// has no stated contract; such a reference is checked for existence only,
    /// which is all the document can support.
    fn contract(&self, value: &str) -> Option<&crate::schema::TensorContract> {
        self.ports
            .and_then(|ports| ports.inputs.get(value).or_else(|| ports.outputs.get(value)))
            .or_else(|| self.contracts.get(value))
    }
}

fn is_integer_dtype(dtype: &str) -> bool {
    matches!(
        dtype,
        "int8" | "int16" | "int32" | "int64" | "uint8" | "uint16" | "uint32" | "uint64"
    )
}

/// Every value name the workflow declares, with the contracts it states directly.
fn workflow_declared_values(
    workflow: &WorkflowSpec,
) -> (
    BTreeSet<String>,
    BTreeMap<String, crate::schema::TensorContract>,
) {
    fn collect_invoke_contracts(
        steps: &[WorkflowStep],
        workflow: &WorkflowSpec,
        contracts: &mut BTreeMap<String, crate::schema::TensorContract>,
    ) {
        for step in steps {
            match step {
                WorkflowStep::Sequence { steps } => {
                    collect_invoke_contracts(steps, workflow, contracts)
                }
                WorkflowStep::Invoke {
                    component, outputs, ..
                } => {
                    let Some(declaration) = workflow.components.get(component) else {
                        continue;
                    };
                    for (port, value) in outputs {
                        if let Some(contract) = declaration.ports.outputs.get(port) {
                            contracts.insert(value.clone(), contract.clone());
                        }
                    }
                }
                WorkflowStep::Loop { setup, steps, .. } => {
                    collect_invoke_contracts(setup, workflow, contracts);
                    collect_invoke_contracts(steps, workflow, contracts);
                }
                WorkflowStep::Branch { cases, default, .. } => {
                    for case in cases.values() {
                        collect_invoke_contracts(std::slice::from_ref(case), workflow, contracts);
                    }
                    if let Some(default) = default {
                        collect_invoke_contracts(
                            std::slice::from_ref(default),
                            workflow,
                            contracts,
                        );
                    }
                }
                WorkflowStep::Emit { .. } => {}
            }
        }
    }

    let mut declared = workflow_step_produced_values(&workflow.steps);
    let mut contracts = BTreeMap::new();
    for (name, input) in &workflow.inputs {
        declared.insert(name.clone());
        contracts.insert(name.clone(), input.contract.clone());
        if let Some(present_as) = &input.present_as {
            declared.insert(present_as.clone());
        }
    }
    for (name, state) in &workflow.state {
        declared.insert(name.clone());
        contracts.insert(name.clone(), state.contract.clone());
    }
    for (name, output) in &workflow.outputs {
        declared.insert(name.clone());
        contracts
            .entry(name.clone())
            .or_insert_with(|| output.contract.clone());
    }
    collect_invoke_contracts(&workflow.steps, workflow, &mut contracts);
    (declared, contracts)
}

/// A packed layout and a validity companion are references, and a reference
/// that resolves to nothing — or to a value that cannot carry what it is asked
/// to carry — is a contract no runtime can execute.
///
/// `token_packed` is the only layout whose meaning lives in other values: the
/// offsets say how many items each owner contributed and the owner map says
/// which owner each item came from. Together they are what lets a runtime split
/// a packed result back into per-request pieces without any serialized row
/// identity, so they must be values that exist, are integers, and are shaped
/// the way that mapping requires. `padding` is the same kind of fact for a
/// padded batch.
fn validate_batch_layout_references(workflow: &WorkflowSpec, errors: &mut Vec<String>) {
    let (declared, contracts) = workflow_declared_values(workflow);
    let workflow_scope = LayoutReferenceScope {
        declared: &declared,
        contracts: &contracts,
        ports: None,
    };
    for (name, input) in &workflow.inputs {
        validate_contract_references(
            &format!("workflow input '{name}'"),
            Some(name),
            &input.contract,
            &workflow_scope,
            errors,
        );
    }
    for (name, output) in &workflow.outputs {
        validate_contract_references(
            &format!("workflow output '{name}'"),
            Some(name),
            &output.contract,
            &workflow_scope,
            errors,
        );
    }
    for (name, state) in &workflow.state {
        validate_contract_references(
            &format!("workflow state '{name}'"),
            Some(name),
            &state.contract,
            &workflow_scope,
            errors,
        );
    }
    for (name, component) in &workflow.components {
        let scope = LayoutReferenceScope {
            declared: &declared,
            contracts: &contracts,
            ports: Some(&component.ports),
        };
        for (direction, ports) in [
            ("input", &component.ports.inputs),
            ("output", &component.ports.outputs),
        ] {
            for (port, contract) in ports {
                let path = format!("workflow component '{name}' {direction} '{port}'");
                validate_contract_references(&path, Some(port), contract, &scope, errors);
                validate_packed_extent(&path, direction, contract, &component.ports, errors);
            }
        }
    }
}

/// Where a packed component port's extent comes from, and who produces the
/// companions that state it.
///
/// An output of the same rank and symbols as its input may be a per-item
/// transform or a token merger, and the two split at completely different
/// boundaries. Nothing in the contract distinguishes them, so the package
/// declares which — and a runtime that had to guess would slice the payload at
/// the wrong places and report nothing at all.
fn validate_packed_extent(
    path: &str,
    direction: &str,
    contract: &crate::schema::TensorContract,
    ports: &crate::schema::ComponentPorts,
    errors: &mut Vec<String>,
) {
    let levels = contract.batch_layout.levels();
    if levels.is_empty() {
        return;
    }
    for (index, level) in levels.iter().enumerate() {
        if direction == "input" {
            if let Some(extent) = level.extent {
                errors.push(format!(
                    "{path} declares level {index} extent {}; every count of a value a component \
                     consumes is one its caller assembled, so only an output states where a level \
                     came from",
                    extent.name()
                ));
            }
            continue;
        }
        let Some(extent) = level.extent else {
            errors.push(format!(
                "{path} packs items but declares no extent for level {index}; a level either \
                 preserves an input level's units one for one or produces its own, and a runtime \
                 that guessed would split the result at the wrong boundaries"
            ));
            continue;
        };
        // Each level answers for itself. The mixed chain — an inner level the
        // graph produced sitting under an outer one it left exactly as it found
        // it — is the ordinary shape of a token-merging encoder, and a single
        // answer for the whole chain could only be wrong at one end.
        for (role, companion) in [("offsets", &level.offsets), ("owner map", &level.owner)] {
            match extent {
                // Preserving a count means reusing the companions that already
                // describe it. A companion the component's own graph emits
                // describes units that did not exist when the call was
                // assembled, so it cannot be the one being preserved.
                crate::schema::PackedExtent::Preserved => {
                    if ports.outputs.contains_key(companion) {
                        errors.push(format!(
                            "{path} declares level {index} extent preserved but its {role} \
                             '{companion}' is an output port of the same component; preserving a \
                             level means reusing the companions that already described it, and \
                             one the graph emits describes units the caller never assembled"
                        ));
                    }
                }
                // A count the graph decides is described by companions the graph
                // emits. Reusing an input's offsets here would describe a length
                // the output does not have, and the split would land between
                // items.
                crate::schema::PackedExtent::Produced => {
                    if !ports.outputs.contains_key(companion) {
                        errors.push(format!(
                            "{path} declares level {index} extent produced but its {role} \
                             '{companion}' is not an output port of the same component; a level \
                             the graph decides is described by companions the graph emits"
                        ));
                    }
                }
            }
        }
        // Correspondence is by the pair, not by the position. An output that
        // consumed its inner level carries the surviving pair at index zero
        // while the input carries it at index one, so matching by index would
        // reject the ordinary token-merging encoder and accept an output that
        // claims to preserve a grouping nothing handed it.
        if extent == crate::schema::PackedExtent::Preserved
            && !ports.inputs.values().any(|input| {
                input.batch_layout.levels().iter().any(|candidate| {
                    candidate.offsets == level.offsets && candidate.owner == level.owner
                })
            })
        {
            errors.push(format!(
                "{path} declares level {index} extent preserved but no input port of the \
                 component declares an ownership level pairing offsets '{}' with owner map '{}'; \
                 a level is preserved by reusing the very pair that described it, and levels \
                 correspond by that pair rather than by their position, since an output may drop \
                 an inner level it consumed",
                level.offsets, level.owner
            ));
        }
    }
}

/// Two packed values that share a level's offsets share its grouping.
///
/// An `offsets` vector is a complete description of how many units each parent
/// owns, so two values naming the same one at the same level are claiming the
/// same grouping. If they pair it with different owner maps, one of the two is
/// packed against a grouping that does not describe it, and a runtime would
/// split whichever it read second at the wrong boundaries.
/// The two counts a companion pair carries, and the site that first stated them.
struct PairExtents<'a> {
    offsets: Option<&'a str>,
    owner: Option<&'a str>,
    first: String,
}

fn validate_shared_companions(workflow: &WorkflowSpec, errors: &mut Vec<String>) {
    let (declared, contracts) = workflow_declared_values(workflow);
    let workflow_scope = LayoutReferenceScope {
        declared: &declared,
        contracts: &contracts,
        ports: None,
    };
    let mut sites: Vec<(
        String,
        &crate::schema::TensorContract,
        LayoutReferenceScope<'_>,
    )> = Vec::new();
    for (name, input) in &workflow.inputs {
        sites.push((
            format!("workflow input '{name}'"),
            &input.contract,
            workflow_scope,
        ));
    }
    for (name, output) in &workflow.outputs {
        sites.push((
            format!("workflow output '{name}'"),
            &output.contract,
            workflow_scope,
        ));
    }
    for (name, state) in &workflow.state {
        sites.push((
            format!("workflow state '{name}'"),
            &state.contract,
            workflow_scope,
        ));
    }
    for (component, spec) in &workflow.components {
        let scope = LayoutReferenceScope {
            declared: &declared,
            contracts: &contracts,
            ports: Some(&spec.ports),
        };
        for (direction, ports) in [
            ("input", &spec.ports.inputs),
            ("output", &spec.ports.outputs),
        ] {
            for (port, contract) in ports {
                sites.push((
                    format!("workflow component '{component}' {direction} '{port}'"),
                    contract,
                    scope,
                ));
            }
        }
    }
    // A level is identified by the pair it names, never by where it sits in a
    // chain. An output that consumed its inner level carries the surviving pair
    // at index zero while its input carries the same pair at index one, so a
    // check keyed on position would call two spellings of one grouping a
    // conflict and would miss a genuine one a level apart.
    let mut pairings: BTreeMap<&str, (&str, String)> = BTreeMap::new();
    let mut owners: BTreeMap<&str, String> = BTreeMap::new();
    let mut extents: BTreeMap<(&str, &str), PairExtents<'_>> = BTreeMap::new();
    for (path, contract, scope) in &sites {
        for level in contract.batch_layout.levels() {
            owners
                .entry(level.owner.as_str())
                .or_insert_with(|| path.clone());
            if let Some((previous, first)) = pairings.get(level.offsets.as_str())
                && *previous != level.owner.as_str()
            {
                errors.push(format!(
                    "{path} pairs offsets '{}' with owner map '{}', but {first} pairs the same \
                     offsets with '{previous}'; one offsets vector describes one grouping, so the \
                     two cannot both be right",
                    level.offsets, level.owner
                ));
                continue;
            }
            pairings.insert(level.offsets.as_str(), (level.owner.as_str(), path.clone()));
            // The pair is a mapping from child units to parent units, so the two
            // numbers it carries — the child count and the parent count plus one
            // — are properties of the mapping and not of the port that names it.
            // A port that resolved either to a different symbol would be reading
            // the same vectors as a different grouping.
            let key = (level.offsets.as_str(), level.owner.as_str());
            let stated = (
                scope.contract(&level.offsets).and_then(extent_symbol),
                scope.contract(&level.owner).and_then(extent_symbol),
            );
            match extents.get(&key) {
                Some(seen) => {
                    let first = &seen.first;
                    for (role, previous, now, companion) in [
                        ("offsets", &seen.offsets, &stated.0, &level.offsets),
                        ("owner map", &seen.owner, &stated.1, &level.owner),
                    ] {
                        if let (Some(previous), Some(now)) = (previous, now)
                            && previous != now
                        {
                            errors.push(format!(
                                "{path} resolves the {role} '{companion}' of the level pairing \
                                 '{}' with '{}' to extent '{now}', but {first} resolves the same \
                                 companion to '{previous}'; a pair is one mapping wherever it is \
                                 named, so its counts cannot differ by the port that names it",
                                level.offsets, level.owner
                            ));
                        }
                    }
                }
                None => {
                    extents.insert(
                        key,
                        PairExtents {
                            offsets: stated.0,
                            owner: stated.1,
                            first: path.clone(),
                        },
                    );
                }
            }
        }
    }
    // An owner map indexes into a batch the caller never sees. Its positions
    // exist only once a group has been formed, and the per-request view of a
    // packed value is derived by rebasing that request's offsets to zero — not
    // by reading an owner vector back. So an owner companion is runtime-internal
    // plumbing, and an application cannot hand one in.
    //
    // This reaches the owner and nothing else. Offsets are per-request
    // meaningful and are delivered rebased, and a `valid_lengths` is already
    // relative to the item it measures — it means the same number in any group,
    // so a request states its own and receives the slice that indexes its own
    // items, with nothing to rebase and no group position to leak.
    for (name, first) in &owners {
        let Some(input) = workflow.inputs.get(*name) else {
            continue;
        };
        if input.externally_suppliable {
            errors.push(format!(
                "workflow input '{name}' is externally_suppliable but {first} names it as an \
                 ownership owner map; an owner map indexes into a batch the application never \
                 sees, and the per-request view is derived by rebasing that request's offsets, so \
                 an owner map is runtime-internal and cannot be supplied"
            ));
        }
    }
}

fn validate_contract_references(
    path: &str,
    value: Option<&str>,
    contract: &crate::schema::TensorContract,
    scope: &LayoutReferenceScope<'_>,
    errors: &mut Vec<String>,
) {
    if let crate::schema::BatchLayout::TokenPacked { axis, levels, .. } = &contract.batch_layout {
        validate_token_packed_layout(path, value, contract, *axis, levels, scope, errors);
    }
    validate_padding(path, value, contract, scope, errors);
}

/// What a packed value's ownership chain must satisfy for a runtime to be able
/// to fold the packed run back into per-request pieces.
///
/// Everything here is a statement about the graph: names resolve, dtypes and
/// ranks can carry what they are asked to carry, and no two declarations
/// contradict each other. What the companions *contain* — that offsets start at
/// zero, rise monotonically, and end at the packed extent, and that every owner
/// position is in range — is a property of the values themselves. Those are
/// checked where the values live, at invocation time on the device that holds
/// them; reading them at load time would mean copying device memory back to the
/// host to answer a question the document cannot answer anyway.
#[allow(clippy::too_many_arguments)]
fn validate_token_packed_layout(
    path: &str,
    value: Option<&str>,
    contract: &crate::schema::TensorContract,
    axis: usize,
    levels: &[crate::schema::OwnershipLevel],
    scope: &LayoutReferenceScope<'_>,
    errors: &mut Vec<String>,
) {
    if axis >= contract.rank {
        errors.push(format!(
            "{path} packs items along axis {axis}, outside its rank {}",
            contract.rank
        ));
        return;
    }
    // A per-request piece of a packed value is a contiguous element window only
    // when the items are the outermost stride. An inner packed axis would make
    // every split a strided gather — a full device-side copy of the payload, per
    // row, per invocation — which is the cost packing exists to avoid. This holds
    // wherever `token_packed` appears, not only where a component declares a
    // capacity: an emit rebases offsets and derives per-request owners with no
    // capacity in sight and wants the same contiguity, so a component that
    // declared none has simply not said it could pay for a gather.
    if axis != 0 {
        errors.push(format!(
            "{path} packs items along axis {axis}; a packed axis must be axis 0, because only \
             then is each request's span a contiguous range that can be aliased rather than \
             gathered"
        ));
    }
    if levels.is_empty() {
        errors.push(format!(
            "{path} packs items but declares no ownership levels; a packed run is only \
             attributable to requests through at least one offsets/owner pair"
        ));
        return;
    }
    if levels.len() > MAX_OWNERSHIP_LEVELS {
        errors.push(format!(
            "{path} declares {} ownership levels, more than the {MAX_OWNERSHIP_LEVELS} a packed \
             value may carry; parts in items in rows is the deepest chain this schema states, and \
             a deeper one is a schema change rather than something a package asserts into \
             existence",
            levels.len()
        ));
        return;
    }
    // Every companion is a different vector with a different length, so one
    // value cannot serve two of these roles. Naming it twice is a contradiction
    // the runtime would resolve silently by reading the wrong lengths.
    let mut seen: BTreeMap<&str, String> = BTreeMap::new();
    for (level, role, companion) in contract.batch_layout.companions() {
        let role = format!("level {level} {role}");
        if let Some(previous) = seen.insert(companion, role.clone())
            && previous != role
        {
            errors.push(format!(
                "{path} names '{companion}' as both its {previous} and its {role}; the two are \
                 different vectors of different lengths, so one value cannot be both"
            ));
        }
    }
    let packed_symbol = contract
        .shape
        .as_ref()
        .and_then(|shape| shape.get(axis))
        .and_then(symbol_of);
    for (index, level) in levels.iter().enumerate() {
        let owner = validate_ownership_level(path, value, index, level, scope, errors);
        // Level zero's owner has one entry per packed position, so the two are
        // the same count and the document has to say so with one symbol. Two
        // symbols is two numbers for one quantity, and a runtime that trusted
        // either would split the payload at the wrong boundaries.
        if index == 0
            && let (Some(packed), Some(owner)) = (packed_symbol, owner.and_then(extent_symbol))
            && packed != owner
        {
            errors.push(format!(
                "{path} packs '{packed}' items on axis {axis} but its level 0 owner map '{}' is \
                 '{owner}' long; the owner map has exactly one entry per packed item, so both \
                 must name the same extent",
                level.owner
            ));
        }
        // A level's offsets is one longer than the count of its parents, which
        // is the count its own owner map indexes into. Reusing the child count
        // there would be an off-by-one the runtime cannot detect.
        if let (Some(offsets), Some(owner)) = (
            scope.contract(&level.offsets).and_then(extent_symbol),
            scope.contract(&level.owner).and_then(extent_symbol),
        ) && offsets == owner
        {
            errors.push(format!(
                "{path} declares level {index} offsets '{}' and owner map '{}' with the same \
                 extent '{offsets}'; offsets carries one entry per parent plus a final total \
                 while the owner map carries one entry per unit, so the two are never equal",
                level.offsets, level.owner
            ));
        }
    }
}

/// Deepest ownership chain a packed value may declare.
///
/// Parts in items in rows — frames in clips in requests, tokens in segments in
/// requests — is what every known workload needs, and each further level
/// multiplies the chain a runtime walks on every split and the corruption cases
/// that have to be tested.
const MAX_OWNERSHIP_LEVELS: usize = 2;

/// Check one level's companions and hand back the owner map's contract.
///
/// Both companions are `shared` rather than `request_aligned`, and that is
/// structural rather than conservative: an exclusive prefix sum is not
/// permutation-followable. Permuting rows does not permute a prefix-offset
/// vector, it invalidates it, so a runtime that compacts rebuilds the chain. A
/// document that labelled either companion request-aligned would be inviting a
/// gather that silently produces nonsense.
fn validate_ownership_level<'a>(
    path: &str,
    value: Option<&str>,
    index: usize,
    level: &crate::schema::OwnershipLevel,
    scope: &'a LayoutReferenceScope<'_>,
    errors: &mut Vec<String>,
) -> Option<&'a crate::schema::TensorContract> {
    let mut owner_contract = None;
    for (role, companion) in [("offsets", &level.offsets), ("owner map", &level.owner)] {
        if Some(companion.as_str()) == value {
            errors.push(format!(
                "{path} names itself as its own level {index} {role}; a packed value and the \
                 vector that describes its packing are different values"
            ));
            continue;
        }
        if !scope.is_declared(companion) {
            errors.push(format!(
                "{path} names '{companion}' as its level {index} {role}, which is not a declared \
                 value or port in that scope"
            ));
            continue;
        }
        let Some(companion_contract) = scope.contract(companion) else {
            continue;
        };
        if role == "owner map" {
            owner_contract = Some(companion_contract);
        }
        if companion_contract.dtype != "int64" {
            errors.push(format!(
                "{path} level {index} {role} '{companion}' is {} but must be int64; offsets and \
                 owner positions are indices, and a narrower or floating type cannot address a \
                 group the runtime has already assembled",
                companion_contract.dtype
            ));
        }
        if companion_contract.rank != 1 {
            errors.push(format!(
                "{path} level {index} {role} '{companion}' has rank {} but must be rank 1; it \
                 carries one entry per unit and nothing else",
                companion_contract.rank
            ));
        }
        if !companion_contract.batch_layout.is_shared() {
            errors.push(format!(
                "{path} level {index} {role} '{companion}' declares {} but must declare shared; \
                 an exclusive prefix sum is not permutation-followable, so a runtime that \
                 compacts rebuilds it rather than gathering it",
                companion_contract.batch_layout.kind_name()
            ));
        }
        if !companion_contract.padding.is_empty() {
            errors.push(format!(
                "{path} level {index} {role} '{companion}' declares padding of its own; a \
                 companion has exactly one entry per unit, so there is nothing in it to pad"
            ));
        }
    }
    owner_contract
}

/// The shape symbol of a dimension, when it has one.
fn symbol_of(dimension: &crate::schema::TensorDimension) -> Option<&str> {
    match dimension {
        crate::schema::TensorDimension::Symbol(symbol) => Some(symbol.as_str()),
        crate::schema::TensorDimension::Fixed(_) => None,
    }
}

/// The extent symbol of a rank-1 companion.
fn extent_symbol(contract: &crate::schema::TensorContract) -> Option<&str> {
    contract
        .shape
        .as_ref()
        .filter(|shape| shape.len() == 1)
        .and_then(|shape| shape.first())
        .and_then(symbol_of)
}

/// Position of a shape symbol in a contract's declared shape.
fn axis_of_symbol(contract: &crate::schema::TensorContract, symbol: &str) -> Option<usize> {
    contract
        .shape
        .as_ref()?
        .iter()
        .position(|dimension| symbol_of(dimension) == Some(symbol))
}

/// What a padded value's validity companions must satisfy.
///
/// Padding is appended, so how much of an entry is real is one number per
/// enclosing position rather than a tensor of booleans. That is what keeps the
/// truth host-resident and cheap: a runtime reads these numbers to assemble and
/// to split every group, and a payload-shaped mask would put that read on the
/// device and turn a free arithmetic check into a hidden transfer.
fn validate_padding(
    path: &str,
    value: Option<&str>,
    contract: &crate::schema::TensorContract,
    scope: &LayoutReferenceScope<'_>,
    errors: &mut Vec<String>,
) {
    let mut covered: BTreeSet<&str> = BTreeSet::new();
    for entry in &contract.padding {
        let crate::schema::PaddedDimension {
            dimension,
            valid_lengths,
        } = entry;
        if !covered.insert(dimension.as_str()) {
            errors.push(format!(
                "{path} declares padding on dimension '{dimension}' more than once; one dimension \
                 has one padded extent, so two companions would be two truths about one fact"
            ));
            continue;
        }
        let Some(axis) = axis_of_symbol(contract, dimension) else {
            errors.push(format!(
                "{path} declares padding on dimension '{dimension}', which is not a shape symbol \
                 of the value it pads"
            ));
            continue;
        };
        // Packed items are contiguous by construction, which is the whole point
        // of packing instead of padding. Padding that same dimension would
        // leave two contradictory accounts of where a unit's entries end.
        if contract.batch_layout.packed_axis() == Some(axis) {
            errors.push(format!(
                "{path} declares padding on dimension '{dimension}', which is the axis it packs \
                 items along; packed items are contiguous and carry no padding, so their extent \
                 is already given by the packing's offsets"
            ));
            continue;
        }
        // A compacted batch has no padding rows: the runtime drops a finished
        // row rather than blanking it. A companion that claimed otherwise would
        // describe a batch the runtime never builds.
        if contract.batch_layout.request_axis() == Some(axis) {
            errors.push(format!(
                "{path} declares padding on dimension '{dimension}', which is the axis its \
                 request rows stack along; padding bounds an extent within a row, and the batch \
                 itself carries no padding rows"
            ));
            continue;
        }
        if Some(valid_lengths.as_str()) == value {
            errors.push(format!(
                "{path} names itself as the valid_lengths of its own dimension '{dimension}'; a \
                 padded value and the vector that bounds it are different values"
            ));
            continue;
        }
        if !scope.is_declared(valid_lengths) {
            errors.push(format!(
                "{path} names '{valid_lengths}' as the valid_lengths of dimension '{dimension}', \
                 which is not a declared value or port in that scope"
            ));
            continue;
        }
        let Some(companion) = scope.contract(valid_lengths) else {
            continue;
        };
        if companion.dtype != "int64" {
            errors.push(format!(
                "{path} valid_lengths '{valid_lengths}' is {} but must be int64; it counts real \
                 entries of dimension '{dimension}'",
                companion.dtype
            ));
        }
        if !companion.batch_layout.is_shared() {
            errors.push(format!(
                "{path} valid_lengths '{valid_lengths}' declares {} but must declare shared; it \
                 has one entry per position of the axes outer to '{dimension}', which is not a \
                 request row count",
                companion.batch_layout.kind_name()
            ));
        }
        validate_valid_lengths_shape(
            path,
            contract,
            dimension,
            axis,
            valid_lengths,
            companion,
            errors,
        );
    }
}

/// A validity companion has exactly one entry per position of the axes outer to
/// the dimension it bounds.
///
/// Axes inner to the padded one are not indexed — a length applies to the whole
/// slice — so the companion's shape is the value's shape truncated at the padded
/// axis. Stating that exactly is what lets a runtime index it without knowing
/// what the value means.
#[allow(clippy::too_many_arguments)]
fn validate_valid_lengths_shape(
    path: &str,
    contract: &crate::schema::TensorContract,
    dimension: &str,
    axis: usize,
    valid_lengths: &str,
    companion: &crate::schema::TensorContract,
    errors: &mut Vec<String>,
) {
    if companion.rank != axis {
        errors.push(format!(
            "{path} valid_lengths '{valid_lengths}' has rank {} but dimension '{dimension}' is \
             axis {axis}, so it must have rank {axis}: one entry per position of the axes outer \
             to '{dimension}'",
            companion.rank
        ));
        return;
    }
    let (Some(outer), Some(declared)) = (contract.shape.as_ref(), companion.shape.as_ref()) else {
        return;
    };
    if outer.len() <= axis || declared.len() != axis {
        return;
    }
    for (index, expected) in outer.iter().take(axis).enumerate() {
        let Some(actual) = declared.get(index) else {
            continue;
        };
        if actual != expected {
            errors.push(format!(
                "{path} valid_lengths '{valid_lengths}' declares {} on axis {index} but the value \
                 it bounds declares {} there; the companion carries one entry per position of the \
                 axes outer to '{dimension}'",
                describe_dimension(actual),
                describe_dimension(expected)
            ));
        }
    }
}

fn describe_dimension(dimension: &crate::schema::TensorDimension) -> String {
    match dimension {
        crate::schema::TensorDimension::Fixed(fixed) => fixed.to_string(),
        crate::schema::TensorDimension::Symbol(symbol) => format!("'{symbol}'"),
    }
}

/// A declared batching capacity is a promise about the artifact's own shape, so
/// it must agree with the ports that carry the group.
///
/// Absence of `batch_capacity` already has a meaning — one request row per
/// invocation — so a declared capacity only ever adds claims: these symbols must
/// already agree, the assembled call materializes no more than this, and every
/// dimension left free is reconciled by a declared padding or packing. A
/// capacity that no port can honour would let a scheduler build an invocation
/// the component cannot execute, which is exactly the kind of contradiction that
/// has to fail at load time.
///
/// Everything is keyed by shape symbol. Ports of one component differ in rank —
/// a rank-3 payload, a rank-1 companion, a rank-2 pooled output — so an axis
/// index would name whichever port the author happened to be looking at, while a
/// symbol names the same quantity on all of them.
fn validate_batch_capacity(workflow: &WorkflowSpec, errors: &mut Vec<String>) {
    for (name, component) in &workflow.components {
        let Some(capacity) = &component.batch_capacity else {
            continue;
        };
        let path = format!("workflow component '{name}' batch_capacity");
        let ports = &component.ports;
        let declared = declared_symbols(ports);
        let counts = ownership_count_symbols(ports);
        let rooted = group_rooted_symbols(ports);
        let budgeted = validate_budgets(&path, capacity, &declared, &rooted, errors);
        validate_uniform_dimensions(&path, capacity, &declared, &counts, errors);
        validate_level_budgets(&path, ports, &budgeted, errors);
        validate_free_dimensions(&path, capacity, ports, errors);
        if let Some(row_scope) = &component.row_scope {
            validate_capacity_row_scope(&path, row_scope, ports, errors);
        }
    }
}

/// Every shape symbol any port of a component declares, with a port that
/// declares it.
///
/// A level's unit count is often named only by its `owner` companion's extent,
/// and a padded extent only by the payload, so a capacity may legitimately name
/// a symbol that appears on exactly one port.
fn declared_symbols(ports: &crate::schema::ComponentPorts) -> BTreeMap<&str, &str> {
    let mut symbols = BTreeMap::new();
    for (port, contract) in ports.inputs.iter().chain(ports.outputs.iter()) {
        let Some(shape) = &contract.shape else {
            continue;
        };
        for dimension in shape {
            if let Some(symbol) = symbol_of(dimension) {
                symbols.entry(symbol).or_insert(port.as_str());
            }
        }
    }
    symbols
}

/// Check the footprint bounds and hand back the symbols they bind.
///
/// A `dimensions` list is a nesting path read outermost first, so `[frames,
/// patches]` bounds "for each packed frame, its materialized patch slots". The
/// first symbol is what roots the entry in the group, which is why it must be a
/// quantity the scheduler chooses rather than a property of one item.
fn validate_budgets<'a>(
    path: &str,
    capacity: &'a crate::schema::ComponentBatchCapacity,
    declared: &BTreeMap<&str, &str>,
    rooted: &BTreeMap<&str, String>,
    errors: &mut Vec<String>,
) -> BTreeSet<&'a str> {
    let mut budgeted: BTreeSet<&str> = BTreeSet::new();
    let mut bounded: BTreeSet<Vec<&str>> = BTreeSet::new();
    for budget in &capacity.budgets {
        let entry = describe_symbols(&budget.dimensions);
        if budget.dimensions.is_empty() {
            errors.push(format!(
                "{path} declares a budgets entry naming no dimension; a bound with nothing to \
                 bind cannot be evaluated against a group"
            ));
            continue;
        }
        if budget.max_total == 0 {
            errors.push(format!(
                "{path} budgets {entry} at 0; every budget is an upper bound on an assembled \
                 group, and a bound of zero forbids the single-item invocation the component is \
                 otherwise required to serve"
            ));
        }
        // Bounding a per-item extent on its own bounds nothing about the
        // invocation being assembled: `patches` alone counts patches per frame,
        // which is a fact about one item and is the same whether the group holds
        // one of them or a thousand.
        if let Some(first) = budget.dimensions.first()
            && declared.contains_key(first.as_str())
            && !rooted.contains_key(first.as_str())
        {
            errors.push(format!(
                "{path} budgets {entry}, whose outermost dimension '{first}' is a property of one \
                 item rather than a count of them; a budget is a nesting path read outermost \
                 first, and one that never reaches a quantity the scheduler chooses bounds \
                 nothing about the group. Root it at {}, or compose it as a path beginning there",
                describe_declared(
                    &rooted
                        .keys()
                        .map(|symbol| (*symbol, "a group count"))
                        .collect::<BTreeMap<_, _>>()
                )
            ));
        }
        let mut within: BTreeSet<&str> = BTreeSet::new();
        for dimension in &budget.dimensions {
            if !within.insert(dimension.as_str()) {
                errors.push(format!(
                    "{path} budgets {entry}, which names '{dimension}' twice; a composed budget \
                     multiplies distinct extents, and squaring one is not a footprint"
                ));
                continue;
            }
            if !declared.contains_key(dimension.as_str()) {
                errors.push(format!(
                    "{path} budgets '{dimension}', which no port of the component declares; \
                     declared symbols are {}",
                    describe_declared(declared)
                ));
                continue;
            }
            budgeted.insert(dimension.as_str());
        }
        let key: Vec<&str> = budget.dimensions.iter().map(String::as_str).collect();
        if !bounded.insert(key) {
            errors.push(format!(
                "{path} budgets {entry} more than once; two bounds on one footprint is two \
                 numbers for one fact, and a runtime honouring either would be honouring neither"
            ));
        }
    }
    budgeted
}

/// Symbols that count what a scheduler put in a group, so a budget rooted at one
/// bounds the invocation rather than one item.
///
/// Three things count a group: the extent a layout packs, the unit count of each
/// declared ownership level — the `owner` companion's extent — and the batch
/// axis of a row-shaped layout, which is how many rows the scheduler put
/// together. A level's `offsets` extent is the parent count plus one, which
/// counts a group too but is a derived spelling of the level above it, so it
/// never roots a budget.
fn group_rooted_symbols(ports: &crate::schema::ComponentPorts) -> BTreeMap<&str, String> {
    let mut rooted: BTreeMap<&str, String> = BTreeMap::new();
    fn axis_symbol(contract: &crate::schema::TensorContract, axis: usize) -> Option<&str> {
        contract
            .shape
            .as_ref()
            .and_then(|shape| shape.get(axis))
            .and_then(symbol_of)
    }
    for (port, contract) in ports.inputs.iter().chain(ports.outputs.iter()) {
        match &contract.batch_layout {
            crate::schema::BatchLayout::TokenPacked { axis, levels } => {
                if let Some(symbol) = axis_symbol(contract, *axis) {
                    rooted
                        .entry(symbol)
                        .or_insert_with(|| format!("the packed extent of port '{port}'"));
                }
                for (index, level) in levels.iter().enumerate() {
                    if let Some(symbol) = ports
                        .inputs
                        .get(&level.owner)
                        .or_else(|| ports.outputs.get(&level.owner))
                        .and_then(extent_symbol)
                    {
                        rooted.entry(symbol).or_insert_with(|| {
                            format!("the units of ownership level {index} of port '{port}'")
                        });
                    }
                }
            }
            // A component that pads rather than packs still assembles a group,
            // and what it assembles is rows. Its row axis is therefore the item
            // count a budget roots at, exactly as a packed extent is.
            crate::schema::BatchLayout::RequestAligned { axis }
            | crate::schema::BatchLayout::RequestExpanded { axis, .. } => {
                if let Some(symbol) = axis_symbol(contract, *axis) {
                    rooted
                        .entry(symbol)
                        .or_insert_with(|| format!("the row count of port '{port}'"));
                }
            }
            _ => {}
        }
    }
    rooted
}

/// Symbols that count units rather than describe one.
///
/// A packed extent, and the extents of the companions of an ownership chain,
/// are exactly the numbers that change when a scheduler forms a group. Each is
/// mapped to the site that made it a count, so a refusal can say which
/// declaration it is arguing with.
fn ownership_count_symbols(ports: &crate::schema::ComponentPorts) -> BTreeMap<&str, String> {
    let mut counts: BTreeMap<&str, String> = BTreeMap::new();
    let resolve = |name: &str| {
        ports
            .inputs
            .get(name)
            .or_else(|| ports.outputs.get(name))
            .and_then(extent_symbol)
    };
    for (port, contract) in ports.inputs.iter().chain(ports.outputs.iter()) {
        let levels = contract.batch_layout.levels();
        if levels.is_empty() {
            continue;
        }
        if let Some(axis) = contract.batch_layout.packed_axis()
            && let Some(symbol) = contract
                .shape
                .as_ref()
                .and_then(|shape| shape.get(axis))
                .and_then(symbol_of)
        {
            counts
                .entry(symbol)
                .or_insert_with(|| format!("the packed extent of port '{port}'"));
        }
        for (index, level) in levels.iter().enumerate() {
            for (role, companion) in [("units", &level.owner), ("run count", &level.offsets)] {
                if let Some(symbol) = resolve(companion) {
                    counts.entry(symbol).or_insert_with(|| {
                        format!("the {role} of ownership level {index} of port '{port}'")
                    });
                }
            }
        }
    }
    counts
}

/// Symbols pinned across a group are stated once and name a property of an item
/// rather than a count of them.
///
/// Pinning is not the opposite of budgeting. A pinned symbol has one extent
/// *within* a group and may differ between groups, so a composed budget that
/// multiplies it by a count is the only thing that bounds the footprint it
/// contributes to; what it may never be is the symbol a layout packs or a
/// level's unit count, because those are what the scheduler chose.
fn validate_uniform_dimensions(
    path: &str,
    capacity: &crate::schema::ComponentBatchCapacity,
    declared: &BTreeMap<&str, &str>,
    counts: &BTreeMap<&str, String>,
    errors: &mut Vec<String>,
) {
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for symbol in &capacity.uniform_dimensions {
        if !seen.insert(symbol.as_str()) {
            errors.push(format!(
                "{path} lists uniform dimension '{symbol}' twice; the list states which extents \
                 must agree, and stating one twice states nothing further"
            ));
            continue;
        }
        if !declared.contains_key(symbol.as_str()) {
            errors.push(format!(
                "{path} requires uniform dimension '{symbol}', which no port of the component \
                 declares; declared symbols are {}",
                describe_declared(declared)
            ));
            continue;
        }
        // Pinning a count is pinning the raggedness away. `uniform_dimensions`
        // says which properties of an *item* must agree for items to share an
        // invocation; how many items a request contributes is the thing a
        // packed layout exists to let vary, and a package that pinned it would
        // be describing a fixed-shape batch it did not declare.
        if let Some(reason) = counts.get(symbol.as_str()) {
            errors.push(format!(
                "{path} requires uniform dimension '{symbol}', which is {reason} rather than a \
                 property of one item; a uniform dimension says what must agree between items for \
                 them to share an invocation, and pinning a count would forbid the very \
                 raggedness the packed layout declares. Bound it with a budget instead, or drop \
                 the ownership level and declare a fixed group"
            ));
        }
    }
}

/// Every ownership level a component's inputs pack is bounded.
///
/// The units of a level are exactly what a scheduler chooses when it forms a
/// group, so a level with no budget leaves the group unbounded in the one
/// quantity the scheduler controls. Output levels are not budgeted: an extent
/// the graph decides cannot be a precondition on forming the group, and the
/// package could not check a bound on it either.
fn validate_level_budgets(
    path: &str,
    ports: &crate::schema::ComponentPorts,
    budgeted: &BTreeSet<&str>,
    errors: &mut Vec<String>,
) {
    for (port, contract) in &ports.inputs {
        for (index, level) in contract.batch_layout.levels().iter().enumerate() {
            let Some(symbol) = ports
                .inputs
                .get(&level.owner)
                .or_else(|| ports.outputs.get(&level.owner))
                .and_then(extent_symbol)
            else {
                continue;
            };
            if !budgeted.contains(symbol) {
                errors.push(format!(
                    "{path} declares no budget for '{symbol}', the units input '{port}' packs at \
                     ownership level {index}; a scheduler chooses how many of those to group, so \
                     an unbudgeted level is unbounded in exactly the quantity it controls"
                ));
            }
        }
    }
}

/// A dimension items may differ on is declared free everywhere, and is
/// reconciled somewhere.
///
/// A fixed literal on a dimension the package has said may vary is a
/// contradiction: the group would have to change a shape the artifact pinned.
/// And a free dimension with neither a padding entry nor a packed axis to
/// consume it is a promise a runtime cannot honour — it would have to invent a
/// reconciliation, which is the silent-wrong-answer class this schema exists to
/// prevent.
fn validate_free_dimensions(
    path: &str,
    capacity: &crate::schema::ComponentBatchCapacity,
    ports: &crate::schema::ComponentPorts,
    errors: &mut Vec<String>,
) {
    let companions = port_companions(ports);
    let uniform: BTreeSet<&str> = capacity
        .uniform_dimensions
        .iter()
        .map(String::as_str)
        .collect();
    for (port, contract) in &ports.inputs {
        if companions.contains(port.as_str()) {
            continue;
        }
        let Some(shape) = &contract.shape else {
            continue;
        };
        let padded: BTreeSet<&str> = contract
            .padding
            .iter()
            .map(|entry| entry.dimension.as_str())
            .collect();
        for (axis, dimension) in shape.iter().enumerate() {
            if contract.batch_layout.request_axis() == Some(axis) {
                continue;
            }
            if contract.batch_layout.packed_axis() == Some(axis) {
                if symbol_of(dimension).is_none() {
                    errors.push(format!(
                        "{path} groups input '{port}', which packs items along axis {axis} but \
                         fixes that axis at {}; a packed extent is the sum of the group's items \
                         and cannot be a literal",
                        describe_dimension(dimension)
                    ));
                }
                continue;
            }
            let Some(symbol) = symbol_of(dimension) else {
                continue;
            };
            if uniform.contains(symbol) || padded.contains(symbol) {
                continue;
            }
            errors.push(format!(
                "{path} leaves '{symbol}' free on input '{port}' axis {axis} but declares neither \
                 a padding entry on it nor a packed axis that consumes it; a dimension items may \
                 differ on has to say how the difference is reconciled"
            ));
        }
    }
}

/// Ports named by another port's layout or padding rather than carrying a
/// payload of their own.
fn port_companions(ports: &crate::schema::ComponentPorts) -> BTreeSet<&str> {
    let mut companions = BTreeSet::new();
    for contract in ports.inputs.values().chain(ports.outputs.values()) {
        for (_, _, companion) in contract.batch_layout.companions() {
            companions.insert(companion);
        }
        for entry in &contract.padding {
            companions.insert(entry.valid_lengths.as_str());
        }
    }
    companions
}

/// Row scope counts rows, and a packed axis counts items.
///
/// One request contributing eight clips makes the two different numbers, so a
/// runtime that compacted per-request state with an item-indexed selection would
/// address the wrong entries entirely. The axis therefore has to be a row axis
/// some port actually declares, and never a packed one.
fn validate_capacity_row_scope(
    path: &str,
    row_scope: &crate::schema::ComponentRowScope,
    ports: &crate::schema::ComponentPorts,
    errors: &mut Vec<String>,
) {
    let mut packed_on: Option<&str> = None;
    let mut rows_on = false;
    for (port, contract) in ports.inputs.iter().chain(ports.outputs.iter()) {
        if contract.batch_layout.request_axis() == Some(row_scope.axis) {
            rows_on = true;
        }
        if contract.batch_layout.packed_axis() == Some(row_scope.axis) && packed_on.is_none() {
            packed_on = Some(port.as_str());
        }
    }
    if let Some(port) = packed_on
        && !rows_on
    {
        errors.push(format!(
            "{path} declares row_scope on axis {}, which port '{port}' packs items along; items \
             are not rows — one request contributes many — so per-row state selected by an item \
             position would address the wrong entries",
            row_scope.axis
        ));
        return;
    }
    if !rows_on {
        errors.push(format!(
            "{path} declares row_scope on axis {}, which no port of the component declares as its \
             request axis; per-row state is selected by row position, so the axis has to be one \
             the component's rows actually stack along",
            row_scope.axis
        ));
    }
}

/// A symbol list, spelled for an error message.
fn describe_symbols(symbols: &[String]) -> String {
    let listed: Vec<String> = symbols.iter().map(|symbol| format!("'{symbol}'")).collect();
    format!("[{}]", listed.join(", "))
}

/// The symbols a component's ports declare, spelled for an error message.
fn describe_declared(declared: &BTreeMap<&str, &str>) -> String {
    let listed: Vec<String> = declared
        .keys()
        .map(|symbol| format!("'{symbol}'"))
        .collect();
    if listed.is_empty() {
        "none".to_string()
    } else {
        listed.join(", ")
    }
}

fn validate_state_lifetimes(workflow: &WorkflowSpec, errors: &mut Vec<String>) {
    // Runtime-owned state is private: its storage layout, paging and precision
    // are the runtime's business and are not part of any contract. Such a cell
    // may only become a package output through its group's versioned checkpoint
    // adapter, which is the single portable, cross-build state path. Private
    // prefill/decode and encoder/decoder transfers are a different mechanism
    // entirely: they are fast because they require a matching runtime protocol
    // and build on both ends, and exporting one as if it were portable is how a
    // rolling upgrade corrupts state. Workflow-owned cells are exempt: they are
    // ordinary typed tensors with a graph-visible representation, so publishing
    // one carries no cross-build hazard.
    let checkpointed_groups = workflow
        .serving
        .as_ref()
        .map(|serving| {
            serving
                .state_service
                .groups
                .iter()
                .filter(|(_, group)| group.checkpoint.is_some())
                .map(|(name, _)| name.as_str())
                .collect::<BTreeSet<_>>()
        })
        .unwrap_or_default();
    // Publication is detected on the emitted value, never on the output key: an
    // emit names an SSA value and an output key that need not match, so keying
    // off the output name lets `emit { value: cache, output: cache_dump }`
    // export runtime-owned state under an alias.
    let state_names = workflow
        .state
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let published = emitted_state_cells(&workflow.steps, &state_names);
    for (name, cell) in &workflow.state {
        let externally_owned = matches!(
            cell.management,
            crate::schema::StateManagement::Runtime | crate::schema::StateManagement::External
        );
        if externally_owned && (workflow.outputs.contains_key(name) || published.contains(name)) {
            let exportable = cell
                .service_group
                .as_deref()
                .is_some_and(|group| checkpointed_groups.contains(group));
            if !exportable {
                errors.push(format!(
                    "pipeline.workflow.state.{name} is published as a package output but its \
                     state group declares no checkpoint adapter; runtime-owned state is private \
                     and leaves the process portably only through a versioned checkpoint"
                ));
            }
        }
        // Binding a cell to a state service group hands its storage to the
        // runtime, so the declared management must say so.
        if cell.service_group.is_some()
            && cell.management != crate::schema::StateManagement::Runtime
        {
            errors.push(format!(
                "pipeline.workflow.state.{name} binds a state service group but is not declared \
                 management: runtime; the group is the runtime's storage"
            ));
        }
        if externally_owned && cell.release_boundary.is_none() {
            errors.push(format!(
                "pipeline.workflow.state.{name} is {} but declares no release_boundary; \
                 externally owned state has no SSA liveness to free it",
                match cell.management {
                    crate::schema::StateManagement::External => "external",
                    _ => "runtime-managed",
                }
            ));
        }
        if !externally_owned && cell.release_boundary.is_some() {
            errors.push(format!(
                "pipeline.workflow.state.{name} declares a release_boundary but is workflow-owned; \
                 workflow state is freed by SSA liveness"
            ));
        }
        if cell.release_boundary == Some(crate::schema::StateReleaseBoundary::Session)
            && cell.scope != crate::schema::WorkflowStateScope::Session
        {
            errors.push(format!(
                "pipeline.workflow.state.{name} releases at a session boundary but is not \
                 session-scoped"
            ));
        }
    }
}

/// A package that claims a conversation survives its invocations must declare
/// what carries it.
///
/// Session scope on its own says only "keep this"; it says nothing about how
/// the next invocation reaches the kept value. Where the document does answer
/// that — a `service_group` naming the runtime's storage, a `continuation`
/// naming the request binding the conversation rejoins — the answer has to
/// resolve, or a package advertises continuity that silently restarts on every
/// turn. That is the failure this rejects, at the document rather than at the
/// third turn of a conversation.
fn validate_session_continuity(workflow: &WorkflowSpec, errors: &mut Vec<String>) {
    let facts = crate::session_state::classify_session_state(workflow);
    let carried = crate::session_state::loop_carried_cells(&workflow.steps);
    let groups = workflow
        .serving
        .as_ref()
        .map(|serving| &serving.state_service.groups);

    let mut continuations = Vec::new();
    for (name, cell) in &workflow.state {
        let path = format!("pipeline.workflow.state.{name}");
        if cell.scope != crate::schema::WorkflowStateScope::Session {
            if cell
                .session
                .as_ref()
                .is_some_and(|lease| lease.continuation.is_some())
            {
                errors.push(format!(
                    "{path} declares a session continuation but is not session-scoped"
                ));
            }
            continue;
        }

        // A session-scoped cell that binds a state service group is that
        // group's storage. Naming a group the document does not declare, or one
        // whose aliases never reach this cell, leaves the lease with nothing to
        // hold — and is why `classify_session_state` does not count such a
        // group as a carrier either.
        if let Some(group_name) = cell.service_group.as_deref() {
            match groups.and_then(|groups| groups.get(group_name)) {
                None => errors.push(format!(
                    "{path} is session-scoped and binds state service group '{group_name}', \
                     which pipeline.workflow.serving.state_service.groups does not declare"
                )),
                Some(group) => {
                    let aliased = group
                        .ports
                        .values()
                        .any(|component| component.contains_key(name));
                    if !aliased {
                        errors.push(format!(
                            "{path} is session-scoped and binds state service group \
                             '{group_name}', but no component alias in that group names it; a \
                             leased cell with no graph port cannot be carried"
                        ));
                    }
                    // A group-backed lease is read where the cell's initializer
                    // is read and written from the alias's `output` port. An
                    // alias with no output names a port the runtime could read
                    // but never advance, so the second turn would replay the
                    // first.
                    let group_is_the_carrier = cell
                        .session
                        .as_ref()
                        .is_none_or(|lease| lease.continuation.is_none())
                        && !carried.contains(name);
                    if group_is_the_carrier
                        && group
                            .ports
                            .values()
                            .filter_map(|component| component.get(name))
                            .any(|alias| alias.output.is_none())
                    {
                        errors.push(format!(
                            "{path} is carried only by state service group '{group_name}', but \
                             an alias for it declares no output port; the lease could be read \
                             and never advanced, so every turn would replay the first"
                        ));
                    }
                    // The lease enters at the alias's `input` port, so some step
                    // has to invoke that component and bind that port. An alias
                    // no step reaches is a lease with no reader.
                    if group_is_the_carrier {
                        let produced = workflow_step_produced_values(&workflow.steps);
                        for (component, alias) in
                            group.ports.iter().filter_map(|(component, aliases)| {
                                aliases.get(name).map(|alias| (component, alias))
                            })
                        {
                            match workflow_component_port_binding(
                                &workflow.steps,
                                component,
                                &alias.input,
                            ) {
                                None => errors.push(format!(
                                    "{path} is carried only by state service group \
                                     '{group_name}', whose alias reads component '{component}' \
                                     port '{}', but no step invokes that component binding that \
                                     port; the lease would have no reader",
                                    alias.input
                                )),
                                // The lease is written to that value before the
                                // pass, which every way of invoking a component
                                // reads — generically, fused into an execution
                                // island, through a host contract, or redirected
                                // by an override. A step that also defines it
                                // would overwrite the lease and restart the
                                // session with no error, so such a package is
                                // refused rather than left to fail quietly.
                                Some(bound) if produced.contains(bound) => errors.push(format!(
                                    "{path} is carried only by state service group \
                                     '{group_name}', but the value '{bound}' its alias reads is \
                                     produced by a step in the same pass, which would overwrite \
                                     the lease; bind the port to a workflow input, or carry the \
                                     cell in the loop"
                                )),
                                Some(_) => {}
                            }
                        }
                    }
                }
            }
        }

        let Some(continuation) = cell
            .session
            .as_ref()
            .and_then(|lease| lease.continuation.as_ref())
        else {
            continue;
        };
        continuations.push(name.clone());

        let crate::schema::SessionContinuation::PromptPrefix {
            prompt_input,
            tokens_output,
        } = continuation;

        if cell.class != crate::schema::WorkflowStateClass::Semantic {
            errors.push(format!(
                "{path} continues a conversation and must be class: semantic; advisory state may \
                 be dropped, and a dropped conversation is a wrong answer rather than a slower one"
            ));
        }
        if cell.management != crate::schema::StateManagement::Runtime {
            errors.push(format!(
                "{path} continues a conversation across invocations, so its storage is the \
                 runtime's and it must declare management: runtime"
            ));
        }
        if cell.release_boundary != Some(crate::schema::StateReleaseBoundary::Session) {
            errors.push(format!(
                "{path} continues a conversation and must declare release_boundary: session; a \
                 conversation released at the invocation it was created in is not one"
            ));
        }
        if carried.contains(name) {
            errors.push(format!(
                "{path} declares a session continuation and is also loop-carried; the lease and \
                 an SSA carry are two answers about the same value"
            ));
        }
        match &cell.recurrence {
            crate::schema::ShapeRecurrence::Bounded { axis, max }
            | crate::schema::ShapeRecurrence::Growing { axis, max, .. } => {
                if *axis != cell.contract.rank.saturating_sub(1) {
                    errors.push(format!(
                        "{path} continues a conversation along axis {axis}, but tokens accumulate \
                         on the final axis of a rank-{} contract",
                        cell.contract.rank
                    ));
                }
                // A continuation is not loop-carried, so its bound never reaches
                // the carry path where a recurrence value is otherwise resolved.
                // The runtime reads it before the pass runs and again when the
                // pass completes, and both reads need a value that exists by
                // then — which is a declared input, not an SSA value some step
                // produces partway through.
                match workflow.inputs.get(max) {
                    None => errors.push(format!(
                        "{path}.recurrence.max names '{max}', which is not a declared workflow \
                         input; a conversation's bound has to be readable before the turn that \
                         would exceed it runs"
                    )),
                    Some(input) => {
                        validate_integer_scalar_contract(
                            &input.contract,
                            &format!("{path}.recurrence.max"),
                            errors,
                        );
                        if !input.required && input.default.is_none() {
                            errors.push(format!(
                                "{path}.recurrence.max names optional input '{max}', which \
                                 declares no default; a bound a request may omit is not a bound"
                            ));
                        }
                    }
                }
            }
            crate::schema::ShapeRecurrence::Invariant => errors.push(format!(
                "{path} continues a conversation but declares recurrence invariant; a \
                 conversation grows with every turn"
            )),
        }

        match workflow.inputs.get(prompt_input) {
            None => errors.push(format!(
                "{path}.session.continuation.prompt_input names '{prompt_input}', which is not a \
                 declared workflow input"
            )),
            Some(input) => {
                let is_prompt_tokens = matches!(
                    &input.role,
                    crate::schema::SemanticInputRole::Runtime { role, .. }
                        if *role == crate::schema::RuntimeInputRole::PromptTokens
                );
                if !is_prompt_tokens {
                    errors.push(format!(
                        "{path}.session.continuation.prompt_input '{prompt_input}' does not carry \
                         the prompt_tokens runtime role, so prefixing it would change an input \
                         whose meaning this document never stated"
                    ));
                }
                if input.contract.dtype != cell.contract.dtype
                    || input.contract.rank != cell.contract.rank
                {
                    errors.push(format!(
                        "{path}.session.continuation.prompt_input '{prompt_input}' has contract \
                         {:?}/rank {} but the cell holds {:?}/rank {}; a prefix must be the same \
                         kind of tensor as what it prefixes",
                        input.contract.dtype,
                        input.contract.rank,
                        cell.contract.dtype,
                        cell.contract.rank
                    ));
                }
            }
        }

        match workflow.outputs.get(tokens_output) {
            None => errors.push(format!(
                "{path}.session.continuation.tokens_output names '{tokens_output}', which is not \
                 a declared workflow output"
            )),
            Some(output) => {
                if output.role != crate::schema::WorkflowOutputRole::Tokens {
                    errors.push(format!(
                        "{path}.session.continuation.tokens_output '{tokens_output}' does not \
                         carry the tokens output role, so what it publishes is not what a \
                         conversation accumulates"
                    ));
                }
                if output.contract.dtype != cell.contract.dtype {
                    errors.push(format!(
                        "{path}.session.continuation.tokens_output '{tokens_output}' publishes \
                         {:?} but the cell holds {:?}",
                        output.contract.dtype, cell.contract.dtype
                    ));
                }
            }
        }
    }

    for cell in facts.uncarried() {
        // Advisory state is droppable by declaration, so a lease nothing reads
        // costs correctness nothing. Semantic state is the conversation.
        if workflow
            .state
            .get(cell)
            .is_some_and(|state| state.class == crate::schema::WorkflowStateClass::Semantic)
        {
            errors.push(format!(
                "pipeline.workflow.state.{cell} is session-scoped and semantic, but no loop \
                 carries it, no state service group holds it, and its lease names no \
                 continuation; nothing in this document says how the next invocation reaches \
                 the value the lease keeps"
            ));
        }
    }

    if continuations.len() > 1 {
        errors.push(format!(
            "pipeline.workflow.state declares {} session continuations ({}); a package has one \
             conversation, and two cells claiming it leaves no answer about which one a turn \
             continues",
            continuations.len(),
            continuations.join(", ")
        ));
    }
}

/// The SSA value some step binds to `component`'s `port`.
fn workflow_component_port_binding<'a>(
    steps: &'a [WorkflowStep],
    component: &str,
    port: &str,
) -> Option<&'a str> {
    fn walk<'a>(step: &'a WorkflowStep, component: &str, port: &str) -> Option<&'a str> {
        match step {
            WorkflowStep::Sequence { steps } => {
                steps.iter().find_map(|step| walk(step, component, port))
            }
            WorkflowStep::Invoke {
                component: invoked,
                inputs,
                ..
            } => (invoked == component)
                .then(|| inputs.get(port).map(String::as_str))
                .flatten(),
            WorkflowStep::Loop { setup, steps, .. } => setup
                .iter()
                .chain(steps)
                .find_map(|step| walk(step, component, port)),
            WorkflowStep::Branch { cases, default, .. } => cases
                .values()
                .find_map(|step| walk(step, component, port))
                .or_else(|| {
                    default
                        .as_ref()
                        .and_then(|step| walk(step, component, port))
                }),
            WorkflowStep::Emit { .. } => None,
        }
    }
    steps.iter().find_map(|step| walk(step, component, port))
}

/// Every SSA value a step defines.
fn workflow_step_produced_values(steps: &[WorkflowStep]) -> BTreeSet<String> {
    fn walk(step: &WorkflowStep, produced: &mut BTreeSet<String>) {
        match step {
            WorkflowStep::Sequence { steps } => steps.iter().for_each(|step| walk(step, produced)),
            WorkflowStep::Invoke { outputs, .. } => produced.extend(outputs.values().cloned()),
            WorkflowStep::Loop {
                setup,
                steps,
                iteration,
                carried,
                ..
            } => {
                produced.extend(iteration.iter().map(|value| value.value.clone()));
                produced.extend(carried.iter().map(|carry| carry.next.clone()));
                setup
                    .iter()
                    .chain(steps)
                    .for_each(|step| walk(step, produced));
            }
            WorkflowStep::Branch {
                cases,
                default,
                outputs,
                ..
            } => {
                produced.extend(outputs.keys().cloned());
                cases.values().for_each(|step| walk(step, produced));
                if let Some(default) = default {
                    walk(default, produced);
                }
            }
            WorkflowStep::Emit { output, .. } => {
                produced.insert(output.clone());
            }
        }
    }
    let mut produced = BTreeSet::new();
    steps.iter().for_each(|step| walk(step, &mut produced));
    produced
}

/// Speculative regions may only contain effects and state that can be undone to
/// the declared maximum proposal width.
///
/// This deliberately checks `speculation_safety`, never the retry class: an
/// idempotent effect can be safely retried and still be impossible to rewind.
fn validate_speculative_rollback(metadata: &InferenceMetadata, errors: &mut Vec<String>) {
    let Some(speculative) = &metadata.speculative else {
        return;
    };
    let Some(workflow) = metadata
        .pipeline
        .as_ref()
        .map(|pipeline| &pipeline.workflow)
    else {
        errors.push(
            "speculative metadata requires pipeline.workflow to declare the proposer and target"
                .to_string(),
        );
        return;
    };
    for (role, component) in [
        ("proposer", &speculative.proposer),
        ("target", &speculative.target),
    ] {
        if !workflow.components.contains_key(component) {
            errors.push(format!(
                "speculative.{role} '{component}' is not a declared workflow component"
            ));
        }
    }
    if let crate::schema::SpeculativeProposalExecution::Chained {
        token_embedding_input,
        logits_output,
        recurrent,
        folded_carry_output,
        folded_carry_seed,
        token_embedding,
    } = &speculative.proposal_execution
        && let Some(proposer) = workflow.components.get(&speculative.proposer)
    {
        if !proposer.ports.inputs.contains_key(token_embedding_input) {
            errors.push(format!(
                "speculative chained proposer input '{token_embedding_input}' is not an input \
                 port of component '{}'",
                speculative.proposer
            ));
        }
        if !proposer.ports.outputs.contains_key(logits_output) {
            errors.push(format!(
                "speculative chained proposer logits '{logits_output}' is not an output port of \
                 component '{}'",
                speculative.proposer
            ));
        }
        if recurrent.is_empty() && folded_carry_output.is_none() {
            errors.push(
                "speculative chained proposal must declare at least one recurrent binding or a \
                 folded_carry_output"
                    .to_string(),
            );
        }
        if let Some(folded) = folded_carry_output
            && !proposer.ports.outputs.contains_key(folded)
        {
            errors.push(format!(
                "speculative chained folded_carry_output '{folded}' is not an output port of \
                 component '{}'",
                speculative.proposer
            ));
        }
        // A folded carry is pinned by three explicit ports so a runtime never
        // infers by convention: the DESTINATION it lands in
        // (`port_bindings.target_hidden_context`, a proposer input port), the
        // carry_0 SOURCE (`folded_carry_seed`, a target output), and the
        // embedding table for the fused input's leading half (`token_embedding`).
        // Each is required when a folded carry is declared.
        if folded_carry_output.is_some() {
            match speculative.port_bindings.get("target_hidden_context") {
                None => errors.push(
                    "speculative chained proposal declares a folded_carry_output but no \
                     port_bindings.target_hidden_context naming the destination proposer input \
                     port the carry lands in"
                        .to_string(),
                ),
                Some(context_port) => {
                    if !proposer.ports.inputs.contains_key(context_port) {
                        errors.push(format!(
                            "speculative port_bindings.target_hidden_context '{context_port}' is \
                             not an input port of proposer component '{}'",
                            speculative.proposer
                        ));
                    } else if context_port != token_embedding_input {
                        // A folded carry re-enters through the fused input's
                        // trailing half, so the DESTINATION port is the fused
                        // `token_embedding_input` itself, never a separate port.
                        errors.push(format!(
                            "speculative port_bindings.target_hidden_context '{context_port}' must \
                             equal the fused token_embedding_input '{token_embedding_input}'; a \
                             folded carry re-enters through the fused input's trailing half, not a \
                             separate proposer input port"
                        ));
                    }
                }
            }
            match folded_carry_seed {
                None => errors.push(
                    "speculative chained proposal declares a folded_carry_output but no \
                     folded_carry_seed naming the target output that seeds carry_0"
                        .to_string(),
                ),
                Some(seed) => match workflow.components.get(&seed.component) {
                    None => errors.push(format!(
                        "speculative folded_carry_seed component '{}' is not a declared workflow \
                         component",
                        seed.component
                    )),
                    Some(seed_component) => {
                        if !seed_component.ports.outputs.contains_key(&seed.output) {
                            errors.push(format!(
                                "speculative folded_carry_seed output '{}' is not an output port \
                                 of component '{}'",
                                seed.output, seed.component
                            ));
                        }
                        // carry_0 is the target's OWN per-token hidden output, so
                        // the seed must name the speculative target — a proposer
                        // (or any non-target) seed is nonsensical and rejected.
                        if seed.component != speculative.target {
                            errors.push(format!(
                                "speculative folded_carry_seed component '{}' must be the \
                                 speculative target '{}'; the folded carry's first-step seed is \
                                 the target's own hidden output",
                                seed.component, speculative.target
                            ));
                        }
                    }
                },
            }
            match token_embedding {
                None => errors.push(
                    "speculative chained proposal declares a folded_carry_output but no \
                     token_embedding naming where embed(last_token) is gathered from"
                        .to_string(),
                ),
                Some(embedding) => match workflow.components.get(&embedding.component) {
                    None => errors.push(format!(
                        "speculative token_embedding component '{}' is not a declared \
                         workflow component",
                        embedding.component
                    )),
                    Some(embedding_component) => {
                        // The fused input's leading half is `embed(last_token)`
                        // gathered from the TARGET model's shared embedding, so
                        // the table must resolve to a real initializer in the
                        // named target model/artifact. Enforce all three facts
                        // fail-closed: it is the target, the target is an ONNX
                        // model that owns initializers, and the table is named.
                        if embedding.component != speculative.target {
                            errors.push(format!(
                                "speculative token_embedding component '{}' must be the \
                                 speculative target '{}'; a folded carry reuses the target \
                                 model's embedding table",
                                embedding.component, speculative.target
                            ));
                        }
                        let names_onnx_artifact = matches!(
                            &embedding_component.implementation,
                            crate::schema::ComponentImplementation::Onnx { artifact }
                                if !artifact.trim().is_empty()
                        );
                        if !names_onnx_artifact {
                            errors.push(format!(
                                "speculative token_embedding names table '{}' on component '{}', \
                                 which declares no ONNX model artifact for that initializer to \
                                 resolve against",
                                embedding.table, embedding.component
                            ));
                        }
                        if embedding.table.trim().is_empty() {
                            errors.push(
                                "speculative token_embedding table must name a real target \
                                 initializer, not an empty string"
                                    .to_string(),
                            );
                        }
                        // A normalizer is a positive, finite factor. Zero,
                        // negative, NaN and infinity all produce a table the
                        // proposer reads without complaint and drafts nothing
                        // useful from -- the exact silent failure this field was
                        // added to remove, reintroduced through its own value.
                        if let Some(scale) = embedding.scale
                            && !(scale.is_finite() && scale > 0.0)
                        {
                            errors.push(format!(
                                "speculative token_embedding.scale is {scale}; a normalizer the \
                                 target applies to an embedding row must be finite and positive, \
                                 and a package that applies none omits the field rather than \
                                 declaring 0"
                            ));
                        }
                    }
                },
            }
        }
        // Every state-service port map the proposer owns, across all groups. A
        // recurrence must resolve to a read_write alias in one of these.
        let proposer_group_aliases = workflow
            .serving
            .as_ref()
            .map(|serving| &serving.state_service.groups)
            .into_iter()
            .flat_map(|groups| groups.values())
            .filter_map(|group| group.ports.get(&speculative.proposer))
            .collect::<Vec<_>>();
        let mut states = BTreeSet::new();
        for binding in recurrent {
            if !states.insert(&binding.state) {
                errors.push(format!(
                    "speculative chained proposal repeats recurrent state '{}'",
                    binding.state
                ));
            }
            if !workflow.state.contains_key(&binding.state) {
                errors.push(format!(
                    "speculative chained proposal references unknown recurrent state '{}'",
                    binding.state
                ));
            }
            if !speculative.rollback_state.contains(&binding.state) {
                errors.push(format!(
                    "speculative chained recurrent state '{}' must be listed in rollback_state",
                    binding.state
                ));
            }
            if !proposer.ports.inputs.contains_key(&binding.input) {
                errors.push(format!(
                    "speculative chained recurrence input '{}' is not an input port of component '{}'",
                    binding.input, speculative.proposer
                ));
            }
            if !proposer.ports.outputs.contains_key(&binding.output) {
                errors.push(format!(
                    "speculative chained recurrence output '{}' is not an output port of component '{}'",
                    binding.output, speculative.proposer
                ));
            }
            // A recurrence is a genuine state transition, so it must resolve to a
            // read_write state-service alias on the proposer that advances this
            // exact cell through the same input/output ports the binding names. A
            // missing, read_only, or mismatched alias means the loop carry would
            // never be persisted between proposer invocations — reject it
            // fail-closed rather than silently dropping the recurrence.
            match proposer_group_aliases
                .iter()
                .find_map(|ports| ports.get(&binding.state))
            {
                None => errors.push(format!(
                    "speculative chained recurrent state '{}' has no state-service alias on \
                     proposer '{}'; a recurrence must resolve through \
                     serving.state_service.groups.*.ports.{}",
                    binding.state, speculative.proposer, speculative.proposer
                )),
                Some(alias) => {
                    if alias.access != crate::schema::StatePortAccess::ReadWrite {
                        errors.push(format!(
                            "speculative chained recurrent state '{}' resolves to a read_only \
                             state-service alias on proposer '{}', but a recurrence must advance \
                             the state (read_write)",
                            binding.state, speculative.proposer
                        ));
                    }
                    if alias.input != binding.input {
                        errors.push(format!(
                            "speculative chained recurrent state '{}' binds input '{}' but its \
                             state-service alias names input '{}'",
                            binding.state, binding.input, alias.input
                        ));
                    }
                    match alias.output.as_deref() {
                        Some(output) if output == binding.output => {}
                        Some(output) => errors.push(format!(
                            "speculative chained recurrent state '{}' binds output '{}' but its \
                             state-service alias names output '{}'",
                            binding.state, binding.output, output
                        )),
                        None => errors.push(format!(
                            "speculative chained recurrent state '{}' binds output '{}' but its \
                             state-service alias declares no output",
                            binding.state, binding.output
                        )),
                    }
                }
            }
        }
    }
    let declared_groups = workflow
        .serving
        .as_ref()
        .map(|serving| &serving.state_service.groups);
    // rollback_state names workflow state cells: state is SSA-visible at the
    // cell level, so that is the unit a rejected proposal is undone at. A cell
    // bound to a runtime state group inherits that group's rollback bound, and
    // cascading groups must roll back together.
    let proposal_width = speculative.max_proposal_width;
    let mut required_groups = BTreeSet::new();
    for cell_name in &speculative.rollback_state {
        let Some(cell) = workflow.state.get(cell_name) else {
            errors.push(format!(
                "speculative.rollback_state references unknown workflow state '{cell_name}'"
            ));
            continue;
        };
        if let Some(group) = &cell.service_group {
            required_groups.insert(group.clone());
        }
    }
    let mut pending = required_groups.iter().cloned().collect::<Vec<_>>();
    while let Some(group) = pending.pop() {
        let Some(contract) = declared_groups.and_then(|groups| groups.get(&group)) else {
            errors.push(format!(
                "speculative.rollback_state reaches unknown state service group '{group}'"
            ));
            continue;
        };
        for cascade in &contract.capabilities.cascade {
            if required_groups.insert(cascade.clone()) {
                pending.push(cascade.clone());
            }
        }
    }
    for group in &required_groups {
        let Some(contract) = declared_groups.and_then(|groups| groups.get(group)) else {
            continue;
        };
        match contract.capabilities.rollback_positions {
            None => errors.push(format!(
                "speculative.rollback_state group '{group}' declares no rollback_positions; a \
                 rejected proposal could not be undone"
            )),
            Some(positions) if positions < proposal_width => errors.push(format!(
                "speculative.rollback_state group '{group}' rolls back {positions} positions but \
                 the declared maximum proposal width is {proposal_width}"
            )),
            Some(_) => {}
        }
    }
    // Only components that actually execute speculatively are constrained. The
    // check reads speculation_safety, never the retry class: an idempotent
    // effect can be safe to retry and still impossible to rewind.
    //
    // The region is the whole enclosing loop body, not just the two named
    // components. A grammar sidecar or a routing head invoked between the
    // proposer and the target runs on every speculated position too, and an
    // unrewindable effect there is just as fatal to a rejected proposal.
    let speculative_components =
        speculative_region_components(&workflow.steps, &speculative.proposer, &speculative.target);
    let speculative_domains = speculative_components
        .iter()
        .filter_map(|component| workflow.components.get(component.as_str()))
        .flat_map(|component| component.effects.iter().cloned())
        .collect::<BTreeSet<_>>();
    for domain in &speculative_domains {
        let Some(contract) = workflow.effects.get(domain) else {
            continue;
        };
        match &contract.speculation_safety {
            crate::schema::SpeculationSafety::None => errors.push(format!(
                "workflow effect '{domain}' runs inside the speculative region but declares \
                 speculation_safety none; a rejected proposal could not be undone"
            )),
            crate::schema::SpeculationSafety::Rewindable { max_depth }
                if *max_depth < proposal_width =>
            {
                errors.push(format!(
                    "workflow effect '{domain}' rewinds {max_depth} positions but the declared \
                     maximum proposal width is {proposal_width}"
                ));
            }
            crate::schema::SpeculationSafety::Clonable
            | crate::schema::SpeculationSafety::Rewindable { .. } => {}
        }
    }
}

// Recursive validation threads each independent symbol/effect table explicitly.
fn validate_emit_batch_layout_consistency(
    node: &WorkflowNode,
    value_contracts: &BTreeMap<String, crate::schema::TensorContract>,
    outputs: &mut BTreeMap<String, crate::schema::BatchLayout>,
    errors: &mut Vec<String>,
) {
    match node {
        WorkflowNode::Sequence { nodes } => {
            for node in nodes {
                validate_emit_batch_layout_consistency(node, value_contracts, outputs, errors);
            }
        }
        WorkflowNode::Loop { setup, body, .. } => {
            validate_emit_batch_layout_consistency(setup, value_contracts, outputs, errors);
            validate_emit_batch_layout_consistency(body, value_contracts, outputs, errors);
        }
        WorkflowNode::Branch { cases, default, .. } => {
            for case in cases.values() {
                validate_emit_batch_layout_consistency(case, value_contracts, outputs, errors);
            }
            if let Some(default) = default {
                validate_emit_batch_layout_consistency(default, value_contracts, outputs, errors);
            }
        }
        WorkflowNode::Emit { output, value, .. } => {
            let Some(contract) = value_contracts.get(value) else {
                return;
            };
            let layout = contract.batch_layout.clone();
            if let Some(previous) = outputs.insert(output.clone(), layout.clone())
                && previous != layout
            {
                errors.push(format!(
                    "pipeline.workflow output '{output}' mixes batch layouts across emits; every \
                     emit for one output must agree so the runtime can associate result rows with \
                     requests"
                ));
            }
        }
        WorkflowNode::Invoke { .. }
        | WorkflowNode::Transfer { .. }
        | WorkflowNode::ExecutionIsland { .. } => {}
    }
}

/// Every row-scoped value a serving workflow produces must carry a derivable
/// row axis. Row identities are never serialized, so the declared batch layout
/// is the only thing that lets a runtime compact, split, or drop result rows.
fn validate_compaction_derivability(
    node: &WorkflowNode,
    workflow: &WorkflowSpec,
    value_contracts: &BTreeMap<String, crate::schema::TensorContract>,
    path: &str,
    errors: &mut Vec<String>,
) {
    match node {
        WorkflowNode::Sequence { nodes } => {
            for (index, node) in nodes.iter().enumerate() {
                validate_compaction_derivability(
                    node,
                    workflow,
                    value_contracts,
                    &format!("{path}[{index}]"),
                    errors,
                );
            }
        }
        WorkflowNode::Loop { setup, body, .. } => {
            validate_compaction_derivability(
                setup,
                workflow,
                value_contracts,
                &format!("{path}.setup"),
                errors,
            );
            validate_compaction_derivability(
                body,
                workflow,
                value_contracts,
                &format!("{path}.body"),
                errors,
            );
        }
        WorkflowNode::Branch { cases, default, .. } => {
            for (case, node) in cases {
                validate_compaction_derivability(
                    node,
                    workflow,
                    value_contracts,
                    &format!("{path}.cases.{case}"),
                    errors,
                );
            }
            if let Some(default) = default {
                validate_compaction_derivability(
                    default,
                    workflow,
                    value_contracts,
                    &format!("{path}.default"),
                    errors,
                );
            }
        }
        WorkflowNode::Emit {
            value,
            output,
            valid_length,
            when,
            ..
        } => {
            let Some(declared) = workflow.outputs.get(output) else {
                return;
            };
            // A per-row valid_length or guard makes the emission ragged:
            // different rows contribute different amounts, so the result cannot
            // be one dense tensor. That is exactly the case a runtime must split
            // into per-request rows, and it can only do so from a declared row
            // axis, since row identities are no longer serialized. This is
            // checked from the declared output alone, so an emit whose value
            // contract could not be inferred cannot slip past it.
            if (valid_length.is_some() || when.is_some())
                && declared.contract.batch_layout.request_axis().is_none()
            {
                errors.push(format!(
                    "{path} emits '{value}' into output '{output}' with a per-row valid_length or \
                     guard, but '{output}' does not declare request_aligned; ragged emission needs \
                     a declared row axis so the runtime can associate result rows with requests"
                ));
            }
            validate_emit_length_authority(
                path,
                value,
                output,
                valid_length.as_deref(),
                declared,
                errors,
            );
            validate_packed_emit_companions(path, value, output, declared, workflow, errors);
            validate_padded_emit_companions(path, value, output, declared, workflow, errors);
            let Some(contract) = value_contracts.get(value) else {
                return;
            };
            if declared.contract.batch_layout != contract.batch_layout {
                errors.push(format!(
                    "{path} emits '{value}' into output '{output}' with a different batch_layout; \
                     the emitted value and the declared output must agree"
                ));
            }
            // The carve-out is deliberately narrow: a `shared` emitted value is
            // admitted only when it is one of the int64 vectors some other
            // emitted value's contract names as the description of its own shape
            // — an ownership offsets or owner map, or the valid_lengths of a
            // padded dimension — and only at the rank that naming requires of
            // it. All of that is decidable from the declared outputs alone.
            // Anything else `shared` and rank > 0 is still a per-request result
            // with no way back to a request.
            let expectations = output_companions(workflow);
            let claimed = expectations.get(output.as_str());
            let is_shape_companion = contract.dtype == "int64"
                && claimed.is_some_and(|roles| roles.iter().any(|role| role.admits(contract.rank)));
            if contract.rank > 0
                && matches!(contract.batch_layout, crate::schema::BatchLayout::Shared)
                && workflow.serving.is_some()
                && !is_shape_companion
            {
                // A value that is named as a companion but does not have a
                // companion's shape, or is named only by a declaration nothing
                // writes, gets the reason it was not admitted. The generic
                // message would tell its author to declare a row axis, which is
                // precisely what a companion must not do.
                let unwritten = claimed.map(|roles| {
                    roles
                        .iter()
                        .filter(|role| !role.claimant_emitted)
                        .map(|role| role.claimed_by)
                        .collect::<BTreeSet<_>>()
                });
                if let Some(roles) = claimed
                    && roles.iter().any(|role| role.claimant_emitted)
                {
                    let expected = roles
                        .iter()
                        .filter(|role| role.claimant_emitted)
                        .map(|role| match role.rank {
                            Some(rank) => format!("{} at rank {rank}", role.role.describe()),
                            None => role.role.describe().to_string(),
                        })
                        .collect::<BTreeSet<_>>()
                        .into_iter()
                        .collect::<Vec<_>>()
                        .join(", ");
                    errors.push(format!(
                        "{path} emits '{value}' into output '{output}', which another emitted \
                         output names as {expected}, but it is {} at rank {}; a companion is \
                         admitted into a serving workflow only at the shape the declaration \
                         naming it requires",
                        contract.dtype, contract.rank
                    ));
                } else if let Some(unwritten) = unwritten.filter(|names| !names.is_empty()) {
                    errors.push(format!(
                        "{path} emits '{value}' into output '{output}', which only output '{}' \
                         names as a shape companion, and that output is never emitted; a \
                         companion describes a result the caller receives, so it is admitted into \
                         a serving workflow only alongside the value it describes",
                        unwritten.iter().copied().collect::<Vec<_>>().join("', '")
                    ));
                } else {
                    errors.push(format!(
                        "{path} emits per-request value '{value}' without a declared \
                         batch_layout; a serving workflow must declare request_aligned or \
                         token_packed so the runtime can associate result rows with requests"
                    ));
                }
            }
        }
        WorkflowNode::Invoke { .. }
        | WorkflowNode::Transfer { .. }
        | WorkflowNode::ExecutionIsland { .. } => {}
    }
}

/// What some declared output's contract claims about a value it names as the
/// description of its own shape.
///
/// The three companion kinds do not have the same shape, and a carve-out keyed
/// on the name alone cannot tell them apart. Offsets and owner maps are always
/// rank one: one entry per parent boundary and one per child. A validity length
/// has one entry per position of the axes *outer* to the dimension it bounds, so
/// its rank is that dimension's axis index — rank one for a padded axis 1, rank
/// two for a padded axis 2, and so on. Carrying the expectation beside the name
/// is what lets the carve-out admit each of them at its own shape instead of at
/// whichever shape happened to be written down first.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct CompanionExpectation<'a> {
    role: CompanionRole,
    /// Rank the naming declaration requires, when the declaration determines it.
    ///
    /// `None` where the bounded value's shape does not resolve the padded
    /// dimension to an axis. That is itself an error, reported against the
    /// padding declaration; the carve-out must not add a second, misleading
    /// complaint about a rank nothing was able to compute.
    rank: Option<usize>,
    /// Output whose contract names this companion.
    claimed_by: &'a str,
    /// Whether some step actually emits that output.
    ///
    /// A declaration that is never written describes a result the caller never
    /// receives, so it cannot be the reason another value is admitted. Demanding
    /// that a companion be emitted while letting an unemitted payload confer the
    /// carve-out on it would be two answers to one question.
    claimant_emitted: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum CompanionRole {
    Offsets,
    Owner,
    ValidLengths,
}

impl CompanionRole {
    fn describe(self) -> &'static str {
        match self {
            Self::Offsets => "an ownership level's offsets",
            Self::Owner => "an ownership level's owner map",
            Self::ValidLengths => "a padded dimension's valid_lengths",
        }
    }
}

impl CompanionExpectation<'_> {
    fn admits(self, rank: usize) -> bool {
        self.claimant_emitted && self.rank.is_none_or(|expected| expected == rank)
    }
}

/// Every value a declared output's contract names as a description of its own
/// shape, with the shape that naming requires of it.
///
/// These are `shared` by construction — a prefix-offset vector describes a whole
/// packing, and a length vector describes one entry per position outside the
/// dimension it bounds — so they are exactly the values the per-request emission
/// rule above would otherwise reject. Publishing them is what makes a packed or
/// padded result readable, so the rule that demands a row correspondence has to
/// know which values are the mechanism by which that correspondence is stated.
///
/// A name can be claimed by more than one declaration — one length vector may
/// bound the same dimension of two outputs — so the expectations are collected
/// rather than overwritten, and a value satisfies the carve-out by matching any
/// one of them. Two declarations that demand different ranks of one value are a
/// contradiction, but it is `validate_padding`'s to report against the
/// declaration that is wrong, not this rule's to report as a missing row axis.
///
/// The claiming output must itself be emitted. A declared output no step writes
/// describes a result the caller never receives, so it cannot be the reason
/// another value is admitted — and the companion obligations demand that a
/// companion be emitted, so letting an unemitted payload confer the carve-out
/// would be two answers to one question. The claimant is carried so a value
/// admitted by nothing but an unwritten declaration can be told that, rather
/// than told to declare a row axis.
fn output_companions<'a>(
    workflow: &'a WorkflowSpec,
) -> BTreeMap<&'a str, BTreeSet<CompanionExpectation<'a>>> {
    let emitted = emitted_outputs(workflow);
    let mut companions: BTreeMap<&str, BTreeSet<CompanionExpectation<'_>>> = BTreeMap::new();
    for (name, output) in &workflow.outputs {
        let contract = &output.contract;
        let claimant_emitted = emitted.contains(name.as_str());
        for (_, role, companion) in contract.batch_layout.companions() {
            let role = if role == "offsets" {
                CompanionRole::Offsets
            } else {
                CompanionRole::Owner
            };
            companions
                .entry(companion)
                .or_default()
                .insert(CompanionExpectation {
                    role,
                    rank: Some(1),
                    claimed_by: name.as_str(),
                    claimant_emitted,
                });
        }
        for entry in &contract.padding {
            companions
                .entry(entry.valid_lengths.as_str())
                .or_default()
                .insert(CompanionExpectation {
                    role: CompanionRole::ValidLengths,
                    rank: axis_of_symbol(contract, &entry.dimension),
                    claimed_by: name.as_str(),
                    claimant_emitted,
                });
        }
    }
    companions
}

/// A padded output has one account of its raggedness, and it is the declared one.
///
/// An emit's `valid_length` truncates what the step writes: it is a step-local
/// instruction, invisible in the output contract, and the caller never receives
/// it. A `padding` entry is the opposite — part of the contract the caller reads,
/// naming a length vector that rule 8 requires the workflow to publish. When an
/// emit sets `valid_length` into an output that declares `padding`, the document
/// states how much of the result is real twice, in two places that nothing
/// reconciles, and only one of them reaches whoever has to decode the tensor.
///
/// Two spellings of one fact is the duplicated state RULES.md rule 10 exists to
/// prevent, and here the duplication is worse than untidy: the two can disagree,
/// and the reader that a caller depends on is the one the emit can silently
/// contradict. So the declared `padding` is authoritative and the emit does not
/// also limit its own prefix.
///
/// This needs no version scoping. `padding` is a `1.1` field, so a document that
/// can express this contradiction has already been required to declare `1.1` by
/// [`validate_schema_version`]; an older document has no way to say the thing at
/// all. It is also a rule about values a document *did* state rather than a
/// demand that it state something new, which is the half of the version split in
/// [`validate_emit_axis`] that stays unconditional at every version.
fn validate_emit_length_authority(
    path: &str,
    value: &str,
    output: &str,
    valid_length: Option<&str>,
    declared: &crate::schema::WorkflowOutput,
    errors: &mut Vec<String>,
) {
    let Some(valid_length) = valid_length else {
        return;
    };
    if declared.contract.padding.is_empty() {
        return;
    }
    let declared_padding = declared
        .contract
        .padding
        .iter()
        .map(|entry| {
            format!(
                "'{}' on dimension '{}'",
                entry.valid_lengths, entry.dimension
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    errors.push(format!(
        "{path} emits '{value}' into output '{output}' with valid_length '{valid_length}', but \
         '{output}' declares padding whose valid_lengths {declared_padding} already says how much \
         of it is real; a padded output has one account of its raggedness and it is the declared \
         padding, which is the only one the caller receives, so drop the emit's valid_length"
    ));
}

/// A packed result is only usable by whoever receives it if the companions that
/// describe the packing come with it.
///
/// The caller of a workflow sees its declared outputs and nothing else. A packed
/// output whose offsets and owner maps are internal SSA values is a flat run of
/// items with no way to say which request any of them belongs to, which is the
/// exact failure the packed layout exists to prevent. So every level's
/// companions are declared outputs too, and the serving rule that would
/// otherwise reject them as `shared` admits them for exactly that reason.
fn validate_packed_emit_companions(
    path: &str,
    value: &str,
    output: &str,
    declared: &crate::schema::WorkflowOutput,
    workflow: &WorkflowSpec,
    errors: &mut Vec<String>,
) {
    let emitted = emitted_outputs(workflow);
    for (level, role, companion) in declared.contract.batch_layout.companions() {
        if !workflow.outputs.contains_key(companion) {
            errors.push(format!(
                "{path} emits '{value}' into token_packed output '{output}' whose level {level} \
                 {role} '{companion}' is not itself a declared workflow output; a caller can only \
                 split a packed result with the companions that describe it, so a packed emit \
                 publishes every level of them"
            ));
            continue;
        }
        if !emitted.contains(companion) {
            errors.push(format!(
                "{path} emits '{value}' into token_packed output '{output}' whose level {level} \
                 {role} '{companion}' is declared as a workflow output but never emitted; a \
                 declared output no step writes is delivered empty, so a caller would receive the \
                 packed result and nothing to split it with"
            ));
        }
    }
}

/// A padded result is only readable by whoever receives it if the lengths that
/// say where the padding starts come with it.
///
/// This is the same obligation `validate_packed_emit_companions` states for a
/// packing, for the other way a contract describes a shape the payload does not
/// carry. A padded output whose valid_lengths is an internal SSA value hands the
/// caller a tensor with trailing entries that mean nothing and no way to know
/// how many — and the schema deliberately refuses a payload-shaped validity mask
/// as the alternative, so the length vector is the only account there is.
fn validate_padded_emit_companions(
    path: &str,
    value: &str,
    output: &str,
    declared: &crate::schema::WorkflowOutput,
    workflow: &WorkflowSpec,
    errors: &mut Vec<String>,
) {
    let emitted = emitted_outputs(workflow);
    for entry in &declared.contract.padding {
        if !workflow.outputs.contains_key(&entry.valid_lengths) {
            errors.push(format!(
                "{path} emits '{value}' into output '{output}', which declares padding on \
                 dimension '{}' whose valid_lengths '{}' is not itself a declared workflow \
                 output; a caller can only tell a padded result's real entries from its padding \
                 with the lengths that bound them, so a padded emit publishes them",
                entry.dimension, entry.valid_lengths
            ));
            continue;
        }
        if !emitted.contains(entry.valid_lengths.as_str()) {
            errors.push(format!(
                "{path} emits '{value}' into output '{output}', which declares padding on \
                 dimension '{}' whose valid_lengths '{}' is declared as a workflow output but \
                 never emitted; a declared output no step writes is delivered empty, so a caller \
                 would receive the padded result and no account of its padding",
                entry.dimension, entry.valid_lengths
            ));
        }
    }
}

/// Outputs some step of the workflow actually writes.
///
/// Declaring an output and never emitting into it are different facts, and only
/// the second one delivers anything. A companion obligation satisfied by the
/// declaration alone would be satisfied by a package that hands its caller a
/// ragged payload and an empty vector, which is the failure the obligation
/// exists to prevent.
///
/// This is deliberately the whole workflow rather than the path the payload's
/// own emit sits on. Whether two emits reach the caller together depends on a
/// branch predicate, and this crate does not evaluate predicates, so a
/// path-sensitive rule would either reject correct packages whose companion is
/// written in a sibling branch or pretend to a precision it does not have.
/// "Somewhere" is what can be decided soundly from the declared steps, and it
/// catches the case that actually occurs: a companion declared to satisfy the
/// contract and then never produced.
fn emitted_outputs(workflow: &WorkflowSpec) -> BTreeSet<&str> {
    fn walk<'a>(steps: &'a [crate::schema::WorkflowStep], emitted: &mut BTreeSet<&'a str>) {
        for step in steps {
            match step {
                crate::schema::WorkflowStep::Emit { output, .. } => {
                    emitted.insert(output.as_str());
                }
                crate::schema::WorkflowStep::Sequence { steps } => walk(steps, emitted),
                crate::schema::WorkflowStep::Loop { setup, steps, .. } => {
                    walk(setup, emitted);
                    walk(steps, emitted);
                }
                crate::schema::WorkflowStep::Branch { cases, default, .. } => {
                    for case in cases.values() {
                        walk(std::slice::from_ref(case), emitted);
                    }
                    if let Some(default) = default {
                        walk(std::slice::from_ref(default), emitted);
                    }
                }
                crate::schema::WorkflowStep::Invoke { .. } => {}
            }
        }
    }

    let mut emitted = BTreeSet::new();
    walk(&workflow.steps, &mut emitted);
    emitted
}

#[allow(clippy::too_many_arguments)]
fn validate_workflow_node(
    node: &WorkflowNode,
    workflow: &WorkflowSpec,
    version: crate::version::SchemaVersion,
    values: &mut BTreeSet<String>,
    value_contracts: &mut BTreeMap<String, crate::schema::TensorContract>,
    effects: &mut BTreeMap<String, String>,
    effect_tokens: &mut BTreeSet<String>,
    path: &str,
    errors: &mut Vec<String>,
) {
    match node {
        WorkflowNode::Sequence { nodes } => {
            if nodes.is_empty() && !path.ends_with(".setup") {
                errors.push(format!("{path}.nodes must not be empty"));
            }
            for (index, node) in nodes.iter().enumerate() {
                validate_workflow_node(
                    node,
                    workflow,
                    version,
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
            let inferred_onnx_ports = declaration.ports.inputs.is_empty()
                && declaration.ports.outputs.is_empty()
                && matches!(
                    declaration.implementation,
                    crate::schema::ComponentImplementation::Onnx { .. }
                );
            for (port, value) in inputs {
                if !inferred_onnx_ports && !declaration.ports.inputs.contains_key(port) {
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
            if !inferred_onnx_ports {
                for (port, contract) in &declaration.ports.inputs {
                    if !contract.optional && !inputs.contains_key(port) {
                        errors.push(format!("{path}.inputs is missing port '{port}'"));
                    }
                }
            }
            for (port, value) in outputs {
                if !inferred_onnx_ports && !declaration.ports.outputs.contains_key(port) {
                    errors.push(format!("{path}.outputs has unknown port '{port}'"));
                }
                define_workflow_value(value, values, &format!("{path}.outputs.{port}"), errors);
                if let Some(contract) = declaration.ports.outputs.get(port) {
                    value_contracts.insert(value.clone(), contract.clone());
                }
            }
            if !inferred_onnx_ports {
                for (port, contract) in &declaration.ports.outputs {
                    if !contract.optional && !outputs.contains_key(port) {
                        errors.push(format!("{path}.outputs is missing port '{port}'"));
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
            continue_when,
            max_iterations,
            iteration,
            carried,
            effects: loop_effects,
            termination: _,
        } => {
            validate_workflow_node(
                setup,
                workflow,
                version,
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
            if let Some(contract) = value_contracts.get(max_iterations) {
                validate_integer_scalar_contract(
                    contract,
                    &format!("{path}.max_iterations"),
                    errors,
                );
            }
            let mut body_values = values.clone();
            let mut body_contracts = value_contracts.clone();
            let mut body_effects = effects.clone();
            for (domain, merge) in loop_effects {
                match effects.get(domain) {
                    Some(incoming) if incoming == &merge.incoming => {}
                    Some(incoming) => errors.push(format!(
                        "{path}.effects.{domain} consumes '{}', but the incoming token is \
                         '{incoming}'",
                        merge.incoming
                    )),
                    None => errors.push(format!(
                        "{path}.effects.{domain} references undeclared effect"
                    )),
                }
                if !effect_tokens.insert(merge.body_input.clone()) {
                    errors.push(format!(
                        "{path}.effects.{domain}.body_input '{}' is already produced",
                        merge.body_input
                    ));
                }
                body_effects.insert(domain.clone(), merge.body_input.clone());
            }
            if let Some(iteration) = iteration {
                if iteration.contract.dtype != "int64" || !matches!(iteration.contract.rank, 0 | 1)
                {
                    errors.push(format!(
                        "{path}.iteration must declare int64 rank 0 or rank 1, got {} rank {}",
                        iteration.contract.dtype, iteration.contract.rank
                    ));
                }
                match (iteration.contract.rank, iteration.contract.shape.as_ref()) {
                    (0, Some(shape)) if !shape.is_empty() => errors.push(format!(
                        "{path}.iteration scalar contract must have an empty shape"
                    )),
                    (1, Some(shape)) if shape.len() != 1 => errors.push(format!(
                        "{path}.iteration rank-one broadcast contract must have one dimension"
                    )),
                    (1, None) => errors.push(format!(
                        "{path}.iteration rank-one broadcast contract must declare its shape"
                    )),
                    _ => {}
                }
                define_workflow_value(
                    &iteration.value,
                    &mut body_values,
                    &format!("{path}.iteration"),
                    errors,
                );
                body_contracts.insert(iteration.value.clone(), iteration.contract.clone());
            }
            for carry in carried {
                let Some(state) = workflow.state.get(&carry.cell) else {
                    errors.push(format!(
                        "{path}.carried references unknown cell '{}'",
                        carry.cell
                    ));
                    continue;
                };
                let recurrence_values = match &state.recurrence {
                    crate::schema::ShapeRecurrence::Invariant => Vec::new(),
                    crate::schema::ShapeRecurrence::Growing { increment, max, .. } => {
                        vec![("increment", increment), ("max", max)]
                    }
                    crate::schema::ShapeRecurrence::Bounded { max, .. } => {
                        vec![("max", max)]
                    }
                };
                for (name, value) in recurrence_values {
                    let recurrence_path =
                        format!("{path}.carried.{}.recurrence.{name}", carry.cell);
                    require_workflow_value(value, values, &recurrence_path, errors);
                    if let Some(contract) = value_contracts.get(value) {
                        validate_integer_scalar_contract(contract, &recurrence_path, errors);
                    }
                }
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
            require_workflow_value(
                continue_when,
                &body_values,
                &format!("{path}.continue_when"),
                errors,
            );
            if let Some(contract) = body_contracts.get(continue_when) {
                validate_bool_control_contract(contract, &format!("{path}.continue_when"), errors);
            }
            validate_workflow_node(
                body,
                workflow,
                version,
                &mut body_values,
                &mut body_contracts,
                &mut body_effects,
                effect_tokens,
                &format!("{path}.body"),
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
                define_workflow_value(&carry.next, values, &format!("{path}.carried.next"), errors);
                if let Some(contract) = body_contracts.get(&carry.body_output) {
                    value_contracts.insert(carry.next.clone(), contract.clone());
                }
            }
            for (domain, merge) in loop_effects {
                if body_effects.get(domain) != Some(&merge.body_output) {
                    errors.push(format!(
                        "{path}.effects.{domain} body produces '{}', but the final body token is \
                         {:?}",
                        merge.body_output,
                        body_effects.get(domain)
                    ));
                }
                if !effect_tokens.insert(merge.produces.clone()) {
                    errors.push(format!(
                        "{path}.effects.{domain}.produces '{}' is already produced",
                        merge.produces
                    ));
                }
                effects.insert(domain.clone(), merge.produces.clone());
            }
        }
        WorkflowNode::Branch {
            predicate,
            cases,
            default,
            outputs,
            effects: merges,
        } => {
            require_workflow_value(predicate, values, &format!("{path}.predicate"), errors);
            if let Some(contract) = value_contracts.get(predicate) {
                validate_predicate_contract(contract, &format!("{path}.predicate"), errors);
            }
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
                    version,
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
                    version,
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

            for (case, node) in cases {
                for (cell, source) in workflow_state_results(node) {
                    if workflow.state.get(&cell).is_some_and(|state| {
                        state.scope == crate::schema::WorkflowStateScope::Session
                    }) && !outputs
                        .values()
                        .any(|phi| phi.cases.get(case) == Some(&source))
                    {
                        errors.push(format!(
                            "{path}.cases.{case} updates session state '{cell}' to '{source}' \
                             without exporting it through a branch output"
                        ));
                    }
                }
            }
            if let Some(default) = default {
                for (cell, source) in workflow_state_results(default) {
                    if workflow.state.get(&cell).is_some_and(|state| {
                        state.scope == crate::schema::WorkflowStateScope::Session
                    }) && !outputs
                        .values()
                        .any(|phi| phi.default.as_ref() == Some(&source))
                    {
                        errors.push(format!(
                            "{path}.default updates session state '{cell}' to '{source}' \
                             without exporting it through a branch output"
                        ));
                    }
                }
            }

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
                    let Some((case_values, case_contracts, _, _)) = case_scopes.get(case) else {
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
                define_workflow_value(output, values, &format!("{path}.outputs.{output}"), errors);
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
            when,
            valid_length,
            output,
            mode,
            axis,
            effect_name,
            effect,
        } => {
            require_workflow_value(value, values, &format!("{path}.value"), errors);
            validate_emit_axis(
                value_contracts.get(value),
                mode,
                *axis,
                valid_length.is_some(),
                version,
                path,
                errors,
            );
            if let Some(when) = when {
                require_workflow_value(when, values, &format!("{path}.when"), errors);
                if let Some(contract) = value_contracts.get(when) {
                    validate_bool_control_contract(contract, &format!("{path}.when"), errors);
                }
            }
            if let Some(valid_length) = valid_length {
                require_workflow_value(
                    valid_length,
                    values,
                    &format!("{path}.valid_length"),
                    errors,
                );
                if let Some(contract) = value_contracts.get(valid_length) {
                    validate_integer_control_contract(
                        contract,
                        &format!("{path}.valid_length"),
                        errors,
                    );
                }
            }
            // A row-wise guard or prefix length only has meaning when the
            // emitted value declares a request-aligned axis, because that is
            // what the runtime uses to associate result rows with requests.
            let emitted_request_axis = value_contracts
                .get(value)
                .and_then(|contract| contract.batch_layout.request_axis());
            if emitted_request_axis.is_none() {
                for (field, control) in [
                    ("when", when.as_ref()),
                    ("valid_length", valid_length.as_ref()),
                ] {
                    let Some(contract) = control.and_then(|name| value_contracts.get(name)) else {
                        continue;
                    };
                    let singleton = contract.rank == 0
                        || (contract.rank == 1
                            && matches!(
                                contract.shape.as_deref().and_then(|shape| shape.first()),
                                Some(crate::schema::TensorDimension::Fixed(1))
                            ));
                    if !singleton {
                        errors.push(format!(
                            "{path}.{field} is row-wise but {path}.value declares no \
                             request_aligned batch_layout"
                        ));
                    }
                }
            }
            if !workflow.outputs.contains_key(output) {
                errors.push(format!("{path} emits undeclared output '{output}'"));
            } else if let (Some(value_contract), Some(output_contract)) = (
                value_contracts.get(value),
                workflow.outputs.get(output).map(|output| &output.contract),
            ) {
                if valid_length.is_some() {
                    require_emit_prefix_contracts(
                        value_contract,
                        output_contract,
                        &format!("{path}.output"),
                        errors,
                    );
                } else {
                    require_compatible_contracts(
                        value_contract,
                        output_contract,
                        &format!("{path}.output"),
                        errors,
                    );
                }
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
        WorkflowNode::ExecutionIsland { .. } => {}
    }
}

/// An incremental emit grows one axis of the output, and which axis that is has
/// to be knowable from the document.
///
/// The default - the final axis - is right for a token sequence and wrong for
/// anything whose growth axis sits inside the shape. A rank-four or deeper value
/// is the shape a media tensor takes, `[batch, channels, frames, height, width]`
/// or `[batch, frames, height, width]`, where the final axis is a spatial extent
/// that must never be concatenated. Rather than guess which of several plausible
/// axes was meant, such an emit names it.
///
/// Naming it is a requirement of the version that introduced video, not a
/// retroactive judgement on the documents written before it. A diffusion package
/// that appends latents along a rank-four final axis is correct and always was:
/// the default is what it meant, and it had no way to say so explicitly because
/// nothing asked. Refusing it now would be this crate deciding that shipped
/// packages became invalid the day a video model was added. So the demand starts
/// at [`BATCHING_SCHEMA_VERSION`](crate::version::BATCHING_SCHEMA_VERSION) —
/// which is also the first version in which the ambiguity is real, since a
/// rank-four media tensor is what that version introduced — and an older
/// document keeps the final-axis default it was written against. An explicitly
/// named axis is checked against the rank at every version.
///
/// The general shape of that split is worth stating, because the two halves look
/// alike in a validator and are opposites in effect: a version may gate what a
/// document must *say*, but it never gates what a document may say *wrongly*. A
/// demand to state intent explicitly starts at the version that made the default
/// ambiguous, because a document written earlier meant the default it was
/// written against. Range checks, well-formedness, and every rule about a value
/// a document did state stay unconditional at every version, because relaxing
/// those would let an older spelling assert something false rather than merely
/// leave something unsaid.
fn validate_emit_axis(
    contract: Option<&crate::schema::TensorContract>,
    mode: &crate::schema::WorkflowEmitMode,
    axis: Option<usize>,
    length_limited: bool,
    version: crate::version::SchemaVersion,
    path: &str,
    errors: &mut Vec<String>,
) {
    let Some(contract) = contract else {
        return;
    };
    if let Some(axis) = axis {
        if axis >= contract.rank {
            errors.push(format!(
                "{path}.axis is {axis}, outside the rank {} of value it emits",
                contract.rank
            ));
        }
        return;
    }
    if version < crate::version::BATCHING_SCHEMA_VERSION {
        return;
    }
    let incremental = matches!(mode, crate::schema::WorkflowEmitMode::Append) || length_limited;
    if incremental && contract.rank >= 4 {
        errors.push(format!(
            "{path} grows a rank-{} output but names no axis; the default final axis is a \
             spatial extent for a value of this rank, so the axis it grows along must be stated",
            contract.rank
        ));
    }
}

fn validate_integer_scalar_contract(
    contract: &crate::schema::TensorContract,
    path: &str,
    errors: &mut Vec<String>,
) {
    if !matches!(
        contract.dtype.as_str(),
        "int8" | "int16" | "int32" | "int64" | "uint8" | "uint16" | "uint32" | "uint64"
    ) {
        errors.push(format!("{path} must have an integer dtype"));
    }
    match contract.rank {
        0 => {
            if contract
                .shape
                .as_ref()
                .is_some_and(|shape| !shape.is_empty())
            {
                errors.push(format!("{path} rank-zero contract must have shape []"));
            }
        }
        1 => {
            if !matches!(
                contract.shape.as_deref(),
                Some([crate::schema::TensorDimension::Fixed(1)])
            ) {
                errors.push(format!(
                    "{path} rank-one control contract must have static shape [1]"
                ));
            }
        }
        _ => errors.push(format!("{path} must be a scalar or rank-one tensor")),
    }
}

/// Check that a state group's declared update discipline is self-consistent.
///
/// An `indexed_scatter` group is a rectangular buffer whose valid region is a
/// declared prefix rather than its shape. Every fact that makes that region
/// derivable has to be present, or the runtime is left guessing where a row
/// ends — and a guess here is silent corruption, not a failed load.
/// Check one map of scatter control ports against the components it names.
///
/// Destinations and lengths are both rank-1 integer vectors bound to ordinary
/// inputs, so a typo in either produces a graph that loads and then writes to
/// the wrong place. The only defense is checking the name against the component
/// that is supposed to have it.
fn validate_scatter_control_ports(
    group_name: &str,
    role: &str,
    bindings: &std::collections::BTreeMap<String, String>,
    workflow: &crate::schema::WorkflowSpec,
    errors: &mut Vec<String>,
) {
    for (component_name, port) in bindings {
        match workflow.components.get(component_name) {
            Some(component) => {
                // A component that declares no ports at all has them inferred
                // from its artifact, so there is nothing here to check against.
                let inferred_ports = component.ports.inputs.is_empty()
                    && component.ports.outputs.is_empty()
                    && matches!(
                        component.implementation,
                        crate::schema::ComponentImplementation::Onnx { .. }
                    );
                if !inferred_ports && !component.ports.inputs.contains_key(port) {
                    errors.push(format!(
                        "state service group '{group_name}' component '{component_name}' \
                         {role} port '{port}' is not a declared port"
                    ));
                }
            }
            None => errors.push(format!(
                "state service group '{group_name}' binds {role} to unknown component \
                 '{component_name}'"
            )),
        }
    }
}

fn validate_state_update(
    group_name: &str,
    group: &crate::schema::StateGroupContract,
    workflow: &crate::schema::WorkflowSpec,
    errors: &mut Vec<String>,
) {
    match &group.update {
        Some(crate::schema::StateUpdate::Append)
        | Some(crate::schema::StateUpdate::IndexedScatter { .. })
            if group.sequence_axis.is_none() =>
        {
            errors.push(format!(
                "state service group '{group_name}' uses a sequence update but declares no \
                 sequence_axis"
            ));
        }
        Some(crate::schema::StateUpdate::Replace) => {
            if group.sequence_axis.is_some() {
                errors.push(format!(
                    "state service group '{group_name}' uses replace for fixed-size state but \
                     declares sequence_axis"
                ));
            }
            for (cell_name, cell) in &workflow.state {
                if cell.service_group.as_deref() == Some(group_name)
                    && !matches!(cell.recurrence, crate::schema::ShapeRecurrence::Invariant)
                {
                    errors.push(format!(
                        "workflow state '{cell_name}' binds replace-updated group '{group_name}' \
                         but declares a varying shape; replacement state must remain fixed-size"
                    ));
                }
            }
        }
        _ => {}
    }

    let Some(crate::schema::StateUpdate::IndexedScatter {
        write_indices,
        capacity,
        write_indices_ports,
        kv_length_ports,
    }) = &group.update
    else {
        return;
    };

    if group.logical_lengths.is_none() {
        errors.push(format!(
            "state service group '{group_name}' declares an indexed_scatter update but no \
             logical_lengths; the valid prefix of a fixed-capacity buffer is not derivable from \
             its shape"
        ));
    }

    match workflow.state.get(write_indices) {
        Some(cell) => {
            validate_integer_control_contract(
                &cell.contract,
                &format!(
                    "state service group '{group_name}' write_indices state '{write_indices}'"
                ),
                errors,
            );
            if cell.contract.rank != 1 {
                errors.push(format!(
                    "state service group '{group_name}' write_indices state '{write_indices}' \
                     must be rank one with one destination per row"
                ));
            }
            if cell.class != crate::schema::WorkflowStateClass::Semantic {
                errors.push(format!(
                    "state service group '{group_name}' write_indices state '{write_indices}' \
                     must be semantic: a write cursor that is not restored with its buffer would \
                     overwrite live positions"
                ));
            }
        }
        None => errors.push(format!(
            "state service group '{group_name}' references unknown write_indices state \
             '{write_indices}'"
        )),
    }

    let declared_values = workflow
        .inputs
        .keys()
        .chain(workflow.state.keys())
        .cloned()
        .collect::<BTreeSet<_>>();
    require_workflow_value(
        capacity,
        &declared_values,
        &format!("state service group '{group_name}' capacity"),
        errors,
    );
    if let Some(contract) = workflow
        .inputs
        .get(capacity)
        .map(|input| &input.contract)
        .or_else(|| workflow.state.get(capacity).map(|cell| &cell.contract))
    {
        validate_integer_scalar_contract(
            contract,
            &format!("state service group '{group_name}' capacity '{capacity}'"),
            errors,
        );
    }

    for (role, bindings) in [
        ("write_indices", write_indices_ports),
        ("kv_length", kv_length_ports),
    ] {
        validate_scatter_control_ports(group_name, role, bindings, workflow, errors);
    }

    // A component that reads this group's buffers but never receives the
    // destinations cannot be the one scattering into them, so the runtime would
    // have no way to bounds-check the writes it is about to perform.
    for component_name in group.ports.keys() {
        if !write_indices_ports.contains_key(component_name) {
            errors.push(format!(
                "state service group '{group_name}' binds component '{component_name}' to a \
                 fixed-capacity buffer but declares no write_indices port for it; destinations \
                 cannot be recovered from the step graph"
            ));
        }
        // The destinations say where the next write lands; only the length says
        // how much of the buffer is live. A group whose ports carry explicit
        // key/value roles is advertising the per-layer ABI that a direct driver
        // binds positionally, and that ABI is unbindable without the length —
        // `decoder_io()` would return no static cache at all, silently
        // downgrading the package rather than reporting a fault. Roleless
        // groups are exempt: they bound-check scatters for the engine and never
        // claim to expose the driver ABI.
        let advertises_kv_abi = group.ports.get(component_name).is_some_and(|bindings| {
            bindings.values().any(|alias| {
                matches!(
                    alias.role,
                    Some(crate::schema::StatePortRole::Key | crate::schema::StatePortRole::Value)
                )
            })
        });
        if advertises_kv_abi && !kv_length_ports.contains_key(component_name) {
            errors.push(format!(
                "state service group '{group_name}' gives component '{component_name}' \
                 key/value cache ports but declares no kv_length port for it; the valid prefix \
                 of a capacity-sized buffer is not recoverable from its shape, so the cache ABI \
                 it advertises cannot be bound"
            ));
        }
    }

    // A fixed-capacity buffer that also changes shape is a contradiction: the
    // scatter destinations are only meaningful against one constant extent.
    for (cell_name, cell) in &workflow.state {
        if cell.service_group.as_deref() != Some(group_name) {
            continue;
        }
        if cell_name == write_indices || Some(cell_name) == group.logical_lengths.as_ref() {
            continue;
        }
        if !matches!(cell.recurrence, crate::schema::ShapeRecurrence::Invariant) {
            errors.push(format!(
                "workflow state '{cell_name}' binds indexed_scatter group '{group_name}' but \
                 declares a varying shape; a scattered buffer's capacity is constant, and its \
                 length is carried by logical_lengths"
            ));
        }
    }
}

/// Layer indices must be explicit wherever a group binds several ports of the
/// same role.
///
/// Consumers build per-layer lists positionally, and the binding label is a
/// producer-chosen string whose lexicographic order is not the layer order
/// (`layer.10` sorts before `layer.2`). Two transposed caches still have
/// identical shapes and dtypes, so nothing downstream can detect the swap: the
/// model simply produces subtly wrong tokens. Roleless ports are exempt because
/// each alias carries its own input/output pair and is never placed
/// positionally.
fn validate_state_port_layers(
    group_name: &str,
    group: &crate::schema::StateGroupContract,
    errors: &mut Vec<String>,
) {
    for (component_name, bindings) in &group.ports {
        let mut by_role: BTreeMap<crate::schema::StatePortRole, Vec<(&String, Option<usize>)>> =
            BTreeMap::new();
        for (label, alias) in bindings {
            if let Some(role) = alias.role {
                by_role.entry(role).or_default().push((label, alias.layer));
            }
        }
        for (role, aliases) in &by_role {
            if aliases.len() < 2 {
                continue;
            }
            let role = format!("{role:?}").to_lowercase();
            let missing = aliases
                .iter()
                .filter(|(_, layer)| layer.is_none())
                .map(|(label, _)| label.as_str())
                .collect::<Vec<_>>();
            if !missing.is_empty() {
                errors.push(format!(
                    "state service group '{group_name}' binds {} '{role}' ports for component \
                     '{component_name}' but {} declare no layer; per-layer buffers are paired by \
                     position and label order is not layer order",
                    aliases.len(),
                    missing.join(", ")
                ));
                continue;
            }
            let mut seen = BTreeSet::new();
            for (label, layer) in aliases {
                let Some(layer) = layer else { continue };
                if !seen.insert(*layer) {
                    errors.push(format!(
                        "state service group '{group_name}' binds '{label}' to layer {layer} for \
                         component '{component_name}', which another '{role}' port already \
                         claims; one buffer would shadow the other"
                    ));
                }
            }
        }

        let is_attention = matches!(
            group.kind,
            crate::schema::StateKind::FullAttention
                | crate::schema::StateKind::SlidingAttention
                | crate::schema::StateKind::MultiLatentAttention
                | crate::schema::StateKind::CrossAttention
        );
        let key_aliases = by_role.get(&crate::schema::StatePortRole::Key);
        let value_aliases = by_role.get(&crate::schema::StatePortRole::Value);
        if !is_attention || (key_aliases.is_none() && value_aliases.is_none()) {
            continue;
        }

        let layers = |aliases: Option<&Vec<(&String, Option<usize>)>>| {
            aliases
                .into_iter()
                .flatten()
                .map(|(_, layer)| layer.unwrap_or(0))
                .collect::<BTreeSet<_>>()
        };
        let key_layers = layers(key_aliases);
        let value_layers = layers(value_aliases);
        if key_layers != value_layers {
            let missing_values = key_layers
                .difference(&value_layers)
                .copied()
                .collect::<Vec<_>>();
            let missing_keys = value_layers
                .difference(&key_layers)
                .copied()
                .collect::<Vec<_>>();
            errors.push(format!(
                "state service group '{group_name}' component '{component_name}' must bind \
                 exactly one key and one value alias for the same attention layers; key layers \
                 are {key_layers:?}, value layers are {value_layers:?}, missing value layers \
                 are {missing_values:?}, missing key layers are {missing_keys:?}"
            ));
        }
    }
}

fn validate_integer_control_contract(
    contract: &crate::schema::TensorContract,
    path: &str,
    errors: &mut Vec<String>,
) {
    if !is_integer_dtype(&contract.dtype) {
        errors.push(format!("{path} must have an integer dtype"));
    }
    if !matches!(contract.rank, 0 | 1) {
        errors.push(format!("{path} must be a scalar or rank-one tensor"));
    }
}

fn validate_bool_control_contract(
    contract: &crate::schema::TensorContract,
    path: &str,
    errors: &mut Vec<String>,
) {
    if contract.dtype != "bool" {
        errors.push(format!("{path} must have bool dtype"));
    }
    if !matches!(contract.rank, 0 | 1) {
        errors.push(format!("{path} must be a scalar or rank-one row tensor"));
    }
}

fn validate_predicate_contract(
    contract: &crate::schema::TensorContract,
    path: &str,
    errors: &mut Vec<String>,
) {
    if contract.dtype != "bool" && !is_integer_dtype(&contract.dtype) {
        errors.push(format!("{path} must have a bool or integer dtype"));
    }
    if !matches!(contract.rank, 0 | 1) {
        errors.push(format!(
            "{path} must be a scalar or rank-one broadcast tensor"
        ));
    } else if contract.rank == 1
        && !matches!(
            contract.shape.as_deref(),
            Some([crate::schema::TensorDimension::Fixed(1)])
        )
    {
        errors.push(format!(
            "{path} rank-one predicate contract must have static shape [1]"
        ));
    }
}

fn validate_optional_input_guards(
    node: &WorkflowNode,
    input: &str,
    present_as: &str,
    guaranteed_present: bool,
    path: &str,
    errors: &mut Vec<String>,
) {
    let check = |value: &str, value_path: &str, errors: &mut Vec<String>| {
        if value == input && !guaranteed_present {
            errors.push(format!(
                "{value_path} reads optional input '{input}' outside the true case of its \
                     present_as predicate '{present_as}'"
            ));
        }
    };
    match node {
        WorkflowNode::Sequence { nodes } => {
            for (index, node) in nodes.iter().enumerate() {
                validate_optional_input_guards(
                    node,
                    input,
                    present_as,
                    guaranteed_present,
                    &format!("{path}.nodes[{index}]"),
                    errors,
                );
            }
        }
        WorkflowNode::Invoke {
            inputs: invoke_inputs,
            ..
        } => {
            for (port, value) in invoke_inputs {
                check(value, &format!("{path}.inputs.{port}"), errors);
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
            check(continue_when, &format!("{path}.continue_when"), errors);
            check(max_iterations, &format!("{path}.max_iterations"), errors);
            for (index, carry) in carried.iter().enumerate() {
                check(
                    &carry.current,
                    &format!("{path}.carried[{index}].current"),
                    errors,
                );
                check(
                    &carry.body_output,
                    &format!("{path}.carried[{index}].body_output"),
                    errors,
                );
            }
            validate_optional_input_guards(
                setup,
                input,
                present_as,
                guaranteed_present,
                &format!("{path}.setup"),
                errors,
            );
            validate_optional_input_guards(
                body,
                input,
                present_as,
                guaranteed_present,
                &format!("{path}.body"),
                errors,
            );
        }
        WorkflowNode::Branch {
            predicate,
            cases,
            default,
            outputs,
            ..
        } => {
            check(predicate, &format!("{path}.predicate"), errors);
            for (case, node) in cases {
                let case_present =
                    guaranteed_present || (predicate == present_as && case == "true");
                validate_optional_input_guards(
                    node,
                    input,
                    present_as,
                    case_present,
                    &format!("{path}.cases.{case}"),
                    errors,
                );
            }
            if let Some(default) = default {
                validate_optional_input_guards(
                    default,
                    input,
                    present_as,
                    guaranteed_present,
                    &format!("{path}.default"),
                    errors,
                );
            }
            for (output, phi) in outputs {
                for (case, value) in &phi.cases {
                    let case_present =
                        guaranteed_present || (predicate == present_as && case == "true");
                    if value == input && !case_present {
                        errors.push(format!(
                            "{path}.outputs.{output}.cases.{case} reads optional input '{input}' \
                                 outside its presence branch"
                        ));
                    }
                }
                if let Some(value) = &phi.default {
                    check(value, &format!("{path}.outputs.{output}.default"), errors);
                }
            }
        }
        WorkflowNode::Emit {
            value,
            when,
            valid_length,
            ..
        } => {
            check(value, &format!("{path}.value"), errors);
            if let Some(when) = when {
                check(when, &format!("{path}.when"), errors);
            }
            if let Some(valid_length) = valid_length {
                check(valid_length, &format!("{path}.valid_length"), errors);
            }
        }
        WorkflowNode::Transfer {
            input: transfer_input,
            ..
        } => check(transfer_input, &format!("{path}.input"), errors),
        WorkflowNode::ExecutionIsland { .. } => {}
    }
}

fn require_emit_prefix_contracts(
    actual: &crate::schema::TensorContract,
    declared: &crate::schema::TensorContract,
    path: &str,
    errors: &mut Vec<String>,
) {
    if actual.rank == 0 || declared.rank == 0 {
        errors.push(format!(
            "{path} valid_length requires emitted value and output contracts with rank >= 1"
        ));
        return;
    }
    if actual.dtype != declared.dtype || actual.rank != declared.rank {
        errors.push(format!(
            "{path} has incompatible dtype or rank for prefix emission"
        ));
        return;
    }
    let (Some(actual_shape), Some(declared_shape)) = (&actual.shape, &declared.shape) else {
        return;
    };
    let prefix_axis = actual.rank.saturating_sub(1);
    for (axis, (actual, declared)) in actual_shape.iter().zip(declared_shape).enumerate() {
        if axis != prefix_axis
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
                "{path} has incompatible fixed dimension at axis {axis}"
            ));
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
    let dynamic_axis = match recurrence {
        crate::schema::ShapeRecurrence::Growing { axis, .. } if next => Some(*axis),
        crate::schema::ShapeRecurrence::Bounded { axis, .. } => Some(*axis),
        _ => None,
    };
    for (axis, (actual, declared)) in actual_shape.iter().zip(declared_shape).enumerate() {
        if Some(axis) != dynamic_axis
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

/// A component that owns attention state must say which port carries the
/// sequence.
///
/// Every field of the decode ABI is resolved by role, so a component that
/// declares none yields no ABI at all — `decoder_io()` returns `None` and the
/// runtime falls back to inferring ports from shapes. That fallback is the
/// behaviour the canonical form exists to remove, and reaching it by omission
/// is silent: the document validates, the package loads, and the degradation
/// only shows up as wrong ports much later.
///
/// Transcribed `ports.inputs`/`ports.outputs` contracts do not help here and
/// are not required. An ONNX artifact already states its port names, types, and
/// ranks; what it cannot state is which of several same-typed ports is the
/// autoregressive sequence — `input_ids` and `position_ids` are both rank-2
/// `int64`. So the obligation is exactly one line of `ports.roles`, and it is
/// only owed by components the workflow says own attention state.
fn validate_attention_component_declares_sequence_role(
    group_name: &str,
    group: &crate::schema::StateGroupContract,
    workflow: &WorkflowSpec,
    errors: &mut Vec<String>,
) {
    if !crate::decoder_abi::is_self_attention(group.kind) {
        return;
    }
    // Only a package that depends on the single-decoder lowering can be hurt by
    // a missing role. A workflow with several neural components — an
    // encoder-decoder pair, a speculative draft and verifier, a TTS talker and
    // code predictor — executes each one through its own invoke bindings, and
    // `decoder_io()` deliberately declines to nominate one of them as "the"
    // decoder. Those components own attention state without ever being asked
    // for a decode ABI, so demanding a sequence role from them would be
    // demanding a declaration nothing reads.
    let neural_components = workflow
        .components
        .values()
        .filter(|component| {
            matches!(
                component.implementation,
                crate::schema::ComponentImplementation::Onnx { .. }
            )
        })
        .count();
    if neural_components != 1 {
        return;
    }
    for component_name in group.ports.keys() {
        let Some(component) = workflow.components.get(component_name) else {
            continue;
        };
        if !matches!(
            component.implementation,
            crate::schema::ComponentImplementation::Onnx { .. }
        ) {
            continue;
        }
        let declares_sequence = component.ports.roles.values().any(|role| {
            matches!(
                role,
                crate::schema::PortRole::TokenIds | crate::schema::PortRole::InputsEmbeds
            )
        });
        if !declares_sequence {
            errors.push(format!(
                "state service group '{group_name}' gives component '{component_name}' attention \
                 state, and it is the only neural component in the workflow, so the package \
                 depends on resolving a decode ABI from it; it declares no token_ids or \
                 inputs_embeds role, which leaves that ABI empty and silently falls back to \
                 inferring ports from shapes. Add \
                 pipeline.workflow.components.{component_name}.ports.roles mapping the sequence \
                 port to token_ids (or inputs_embeds). Port contracts are optional and do not \
                 substitute: a graph states its port names and types, but not which one carries \
                 the sequence"
            ));
        }
    }
}

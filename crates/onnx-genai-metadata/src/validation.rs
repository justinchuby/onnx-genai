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
                capability::CONTINUOUS_BATCHING.to_string(),
                capability::CONTROL_FLOW_LOOP.to_string(),
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

    validate_model_io_against_workflow(metadata, &mut errors);

    if let Some(pipeline) = &metadata.pipeline
        && let Err(error) = validate_pipeline_spec(pipeline)
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
    validate_preprocessing_workflow(metadata, &mut errors);
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
        // When padding perturbs a row's values there is no reliable way to
        // recover the valid region by re-deriving it from the input length, so
        // the package must publish the per-row length it actually produced.
        if profile.batch_invariance.as_deref() == Some("padding_sensitive")
            && decoding.lengths.is_none()
        {
            errors.push(format!(
                "profiles.{name}.decoding must bind a lengths output role because \
                 profiles.{name}.batch_invariance is 'padding_sensitive'"
            ));
        }
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

/// One preprocessing output binding, viewed independently of its modality.
///
/// Image and audio programs bind processor-local values to typed SSA names
/// under identical rules, so the workflow checks below are written once against
/// this view rather than duplicated per modality.
struct PreprocessingOutputView<'a> {
    name: &'a str,
    dtype: &'a str,
    contract: Option<&'a crate::schema::TensorContract>,
    optional: bool,
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
    validate_preprocessing_program(
        workflow,
        "image",
        "onnx-genai.image-preprocess",
        preprocessing
            .and_then(|spec| spec.image.as_ref())
            .map(|program| {
                program
                    .outputs
                    .iter()
                    .map(|output| PreprocessingOutputView {
                        name: &output.name,
                        dtype: &output.dtype,
                        contract: output.contract.as_ref(),
                        optional: output.optional.unwrap_or(false),
                    })
                    .collect::<Vec<_>>()
            }),
        errors,
    );
    validate_preprocessing_program(
        workflow,
        "audio",
        "onnx-genai.audio-preprocess",
        preprocessing
            .and_then(|spec| spec.audio.as_ref())
            .map(|program| {
                program
                    .outputs
                    .iter()
                    .map(|output| PreprocessingOutputView {
                        name: &output.name,
                        dtype: &output.dtype,
                        contract: output.contract.as_ref(),
                        optional: output.optional.unwrap_or(false),
                    })
                    .collect::<Vec<_>>()
            }),
        errors,
    );
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
    // batch_layout is part of the contract: a preprocessing output that claims a
    // different row correspondence than the port it feeds would let per-request
    // rows drift out of alignment with the rest of the workflow.
    if normalize(&source.dtype) != normalize(&target.dtype)
        || source.rank != target.rank
        || source.shape != target.shape
        || source.batch_layout != target.batch_layout
    {
        errors.push(format!(
            "{path} has a contract incompatible with its adapter output port"
        ));
    }
}

/// Reject a serialized `model.io` beside the workflow that supersedes it.
///
/// `pipeline.workflow` is the canonical serialized expression of a package's
/// executable graph ABI: component ports carry the port inventory, `ports.roles`
/// carry what each port means, and `state_service` groups carry the cache pairs,
/// their aliasing, and how the graph writes into them. Everything an optimized
/// single-graph decode path needs is recoverable from it, so `model.io` beside a
/// workflow is not additional information — it is a second writable answer to a
/// question already answered.
///
/// A second answer is a fork, not redundancy. Nothing forces the two to agree,
/// and when they disagree which one a runtime obeys is decided by whichever code
/// path reached it first. Rejecting the pair is what keeps "the workflow says X"
/// a complete answer.
///
/// An earlier revision permitted the pair and cross-checked it, because a
/// workflow then had nowhere to name the static cache's control ports — two
/// rank-1 integer vectors that are indistinguishable from each other. The
/// missing fact now lives on the binding that consumes it
/// (`update.write_indices_ports` and `update.kv_length_ports`), which removes
/// the reason to keep a second surface at all.
fn validate_model_io_against_workflow(metadata: &InferenceMetadata, errors: &mut Vec<String>) {
    let (Some(model), Some(_)) = (
        metadata.model.as_ref(),
        metadata
            .pipeline
            .as_ref()
            .map(|pipeline| &pipeline.workflow),
    ) else {
        return;
    };
    if model.legacy_io().is_none() {
        return;
    }

    // The workflow is the canonical serialized expression of a package's
    // executable graph ABI, and a second serialized expression of the same
    // facts is not redundancy but a fork: the moment the two disagree, which
    // one a runtime obeys is decided by whichever code path reached it first.
    // Rejecting the pair is what keeps "the workflow says X" a complete answer.
    errors.push(
        "model.io and pipeline.workflow both declare this package's executable graph ABI; the \
         workflow is canonical, so remove model.io and declare ports at \
         pipeline.workflow.components.<component>.ports (with ports.roles for token_ids, \
         inputs_embeds, attention_mask, position_ids, logits, hidden_states, \
         encoder_hidden_states, and audio_features), model state at \
         pipeline.workflow.serving.state_service.groups, and fixed-capacity writes at that \
         group's update.kind: indexed_scatter"
            .to_string(),
    );
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
    validate_workflow(&spec.workflow, &mut errors);
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

fn validate_workflow(workflow: &WorkflowSpec, errors: &mut Vec<String>) {
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
                    if !inferred_ports && !component.ports.outputs.contains_key(&alias.output) {
                        errors.push(format!(
                            "state service group '{group_name}' component '{component_name}' \
                             output alias '{}' is not a declared port",
                            alias.output
                        ));
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
                },
            );
        }
    }
    let mut effects = compiled.initial_effects.clone();
    let mut effect_tokens = effects.values().cloned().collect::<BTreeSet<_>>();
    validate_workflow_node(
        &compiled.graph,
        workflow,
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
    validate_state_lifetimes(workflow, errors);
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

/// State the runtime or an external service owns must say when it may be freed.
///
/// An ordinary tensor's lifetime is SSA liveness: the runtime frees it when
/// nothing can read it again. Runtime-managed and external state has no such
/// bound — nothing in the dataflow says the last reader has run — so the
/// document must name the boundary at which it becomes releasable.
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
            let Some(contract) = value_contracts.get(value) else {
                return;
            };
            if declared.contract.batch_layout != contract.batch_layout {
                errors.push(format!(
                    "{path} emits '{value}' into output '{output}' with a different batch_layout; \
                     the emitted value and the declared output must agree"
                ));
            }
            if contract.rank > 0
                && matches!(contract.batch_layout, crate::schema::BatchLayout::Shared)
                && workflow.serving.is_some()
            {
                errors.push(format!(
                    "{path} emits per-request value '{value}' without a declared batch_layout; a \
                     serving workflow must declare request_aligned or token_packed so the runtime \
                     can associate result rows with requests"
                ));
            }
        }
        WorkflowNode::Invoke { .. }
        | WorkflowNode::Transfer { .. }
        | WorkflowNode::ExecutionIsland { .. } => {}
    }
}

#[allow(clippy::too_many_arguments)]
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
            if nodes.is_empty() && !path.ends_with(".setup") {
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
            effect_name,
            effect,
            ..
        } => {
            require_workflow_value(value, values, &format!("{path}.value"), errors);
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
    if !matches!(
        contract.dtype.as_str(),
        "int8" | "int16" | "int32" | "int64" | "uint8" | "uint16" | "uint32" | "uint64"
    ) {
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
    if contract.dtype != "bool"
        && !matches!(
            contract.dtype.as_str(),
            "int8" | "int16" | "int32" | "int64" | "uint8" | "uint16" | "uint32" | "uint64"
        )
    {
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

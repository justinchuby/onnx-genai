//! Canonical workflow execution-capability admission.

use onnx_genai_metadata::{
    CandidateTreeTopology, ComponentImplementation, DFlashStructure, InferenceMetadata,
    SemanticInputRole, SpeculativeAcceptedPath, SpeculativeContract, SpeculativeProposalExecution,
    StatePortAccess, WorkflowOutputRole, WorkflowSpec, WorkflowStep,
    extensions::{DFLASH_FLAT_BLOCK_V1, find},
};

use crate::engine::{EngineDecodeBackend, PackageExecutionError};

type InvocationBindings<'a> = (
    &'a std::collections::BTreeMap<String, String>,
    &'a std::collections::BTreeMap<String, String>,
);

/// How the admitted driver supplies the target's topology input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CandidateTreeTopologyInput {
    /// An ancestor-mask proposer output flows directly to the target.
    ProposerValue { value: String },
    /// Parent indices are the authored topology; the driver derives the
    /// equivalent ancestor mask from that exact value.
    DerivedFromParentIndices {
        topology_value: String,
        placeholder: String,
    },
}

/// Authored control-flow frames that determine whether the candidate seam runs.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum CandidateTreeControlFrame {
    LoopSetup { path: String },
    LoopBody { path: String },
    BranchCase { path: String, key: String },
    BranchDefault { path: String },
}

/// Whether one authored entry drains the request budget or advances one block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CandidateTreeExecutionMode {
    /// A flat/root/setup seam is the whole speculative operation.
    DrainAtSeam,
    /// An authored loop owns repetition and invokes the seam once per body entry.
    OncePerAuthoredEntry,
}

/// Immutable candidate-tree bindings proved before component loading.
///
/// Execution consumes this value rather than resolving protocol ports or SSA
/// values again. That keeps admission and execution from answering the same
/// identity question differently.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CandidateTreeExecutionPlan {
    pub(crate) version: String,
    pub(crate) proposer: String,
    pub(crate) target: String,
    pub(crate) proposer_path: String,
    pub(crate) target_path: String,
    pub(crate) control_provenance: Vec<CandidateTreeControlFrame>,
    pub(crate) execution_mode: CandidateTreeExecutionMode,
    pub(crate) topology: CandidateTreeTopology,
    pub(crate) accepted_path_binding: String,
    pub(crate) proposer_bindings: std::collections::BTreeMap<String, String>,
    pub(crate) proposer_outputs: std::collections::BTreeMap<String, String>,
    pub(crate) target_bindings: std::collections::BTreeMap<String, String>,
    pub(crate) target_outputs: std::collections::BTreeMap<String, String>,
    pub(crate) proposer_context_input: String,
    pub(crate) target_context_input: String,
    pub(crate) target_candidate_input: String,
    pub(crate) target_candidate_value: String,
    pub(crate) target_topology_input: String,
    pub(crate) target_topology_value: CandidateTreeTopologyInput,
    pub(crate) target_position_input: String,
    pub(crate) target_position_placeholder: String,
    pub(crate) target_accepted_input: String,
    pub(crate) target_accepted_placeholder: String,
    pub(crate) target_logits_value: String,
    pub(crate) proposal_probabilities_value: Option<String>,
    pub(crate) target_probabilities_value: Option<String>,
    pub(crate) rollback_state: std::collections::BTreeSet<String>,
    pub(crate) max_proposal_width: usize,
    pub(crate) token_output_authority: CandidateTreeTokenOutputAuthority,
}

/// The one S4 publication path for accepted candidate tokens.
///
/// Candidate execution never creates an output independently of the authored
/// workflow. Admission retains the exact canonical output and every source
/// location that proved it consumes the runtime's accepted-path binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CandidateTreeTokenOutputAuthority {
    AuthoredEmit {
        output: String,
        sites: Vec<CandidateTreeTokenEmitSite>,
    },
}

impl CandidateTreeTokenOutputAuthority {
    pub(crate) fn output(&self) -> &str {
        match self {
            Self::AuthoredEmit { output, .. } => output,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CandidateTreeTokenEmitSite {
    pub(crate) path: String,
}

/// The one typed answer to whether this runtime may execute a loaded workflow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum WorkflowExecutionAdmission {
    Admitted,
    CandidateTree(Box<CandidateTreeExecutionPlan>),
    CandidateTreeUnavailable { version: String, reason: String },
    DFlashUnavailable { version: String },
}

impl WorkflowExecutionAdmission {
    pub(crate) fn from_metadata(
        metadata: &InferenceMetadata,
        backend: EngineDecodeBackend,
    ) -> Self {
        Self::from_speculative(
            metadata.speculative.as_ref(),
            metadata
                .pipeline
                .as_ref()
                .map(|pipeline| &pipeline.workflow),
            backend,
        )
    }

    pub(crate) fn from_speculative(
        speculative: Option<&SpeculativeContract>,
        workflow: Option<&onnx_genai_metadata::WorkflowSpec>,
        backend: EngineDecodeBackend,
    ) -> Self {
        let Some(contract) = speculative else {
            return Self::Admitted;
        };
        match &contract.proposal_execution {
            SpeculativeProposalExecution::CandidateTree { .. } => {
                let resolved = if matches!(
                    backend,
                    EngineDecodeBackend::Auto | EngineDecodeBackend::Ort
                ) {
                    resolve_candidate_tree_execution_plan(contract, workflow)
                } else {
                    Err(format!(
                        "backend {backend:?} is unsupported; the production candidate-tree \
                         executor uses the ORT component execution seam"
                    ))
                };
                match resolved {
                    Ok(plan) => Self::CandidateTree(Box::new(plan)),
                    Err(reason) => Self::CandidateTreeUnavailable {
                        version: contract.version.clone(),
                        reason,
                    },
                }
            }
            SpeculativeProposalExecution::DflashFlatBlock {
                version, structure, ..
            } if version == DFLASH_FLAT_BLOCK_V1.version
                && matches!(structure.as_ref(), DFlashStructure::Base)
                && matches!(
                    backend,
                    EngineDecodeBackend::Auto | EngineDecodeBackend::Ort
                ) =>
            {
                Self::Admitted
            }
            SpeculativeProposalExecution::DflashFlatBlock { version, .. } => {
                debug_assert!(
                    find(DFLASH_FLAT_BLOCK_V1.identity, version).is_some(),
                    "metadata validation must reject an unregistered DFlash extension version"
                );
                Self::DFlashUnavailable {
                    version: version.clone(),
                }
            }
            _ => Self::Admitted,
        }
    }

    pub(crate) fn require_supported(&self) -> Result<(), PackageExecutionError> {
        match self {
            Self::Admitted | Self::CandidateTree(_) => Ok(()),
            Self::CandidateTreeUnavailable { version, reason } => {
                Err(PackageExecutionError::CandidateTreeExecutionUnavailable {
                    version: version.clone(),
                    reason: reason.clone(),
                })
            }
            Self::DFlashUnavailable { version } => {
                Err(PackageExecutionError::DFlashExecutionUnavailable {
                    version: version.clone(),
                })
            }
        }
    }

    pub(crate) fn candidate_tree_plan(&self) -> Option<&CandidateTreeExecutionPlan> {
        match self {
            Self::CandidateTree(plan) => Some(plan.as_ref()),
            _ => None,
        }
    }
}

fn resolve_candidate_tree_execution_plan(
    contract: &SpeculativeContract,
    workflow: Option<&WorkflowSpec>,
) -> Result<CandidateTreeExecutionPlan, String> {
    if let Some(reason) = candidate_tree_unavailable_reason(contract, workflow) {
        return Err(reason);
    }
    CandidateTreeExecutionPlan::resolve(
        contract,
        workflow.expect("candidate-tree availability requires a workflow"),
    )
}

impl CandidateTreeExecutionPlan {
    fn resolve(contract: &SpeculativeContract, workflow: &WorkflowSpec) -> Result<Self, String> {
        let SpeculativeProposalExecution::CandidateTree {
            candidate_tokens,
            topology,
        } = &contract.proposal_execution
        else {
            return Err(
                "candidate-tree admission was selected for a non-tree proposal form".to_string(),
            );
        };
        let SpeculativeAcceptedPath::Runtime { binding } = &contract.verification.accepted_path
        else {
            return Err(
                "candidate-tree version 1 requires a runtime accepted-path binding".to_string(),
            );
        };
        let (proposer_bindings, proposer_outputs) =
            component_invocation(&workflow.steps, &contract.proposer).ok_or_else(|| {
                format!(
                    "candidate-tree proposer '{}' has no workflow invocation",
                    contract.proposer
                )
            })?;
        let (target_bindings, target_outputs) =
            component_invocation(&workflow.steps, &contract.target).ok_or_else(|| {
                format!(
                    "candidate-tree target '{}' has no workflow invocation",
                    contract.target
                )
            })?;

        let target_path =
            component_invocation_path(&workflow.steps, &contract.target, "pipeline.workflow.steps")
                .unwrap_or_else(|| "pipeline.workflow.steps".to_string());
        let role = |roles: &std::collections::BTreeMap<String, String>,
                    name: &str,
                    component: &str|
         -> Result<String, String> {
            roles.get(name).cloned().ok_or_else(|| {
                format!(
                    "candidate-tree component '{component}' is missing required protocol role \
                     '{name}'"
                )
            })
        };
        let proposer_context_input = role(
            &contract.port_bindings,
            "context_tokens",
            &contract.proposer,
        )?;
        let target_context_input = role(
            &contract.target_port_bindings,
            "context_tokens",
            &contract.target,
        )?;
        let target_candidate_input = role(
            &contract.target_port_bindings,
            "candidate_tokens",
            &contract.target,
        )?;
        let target_topology_input = role(
            &contract.target_port_bindings,
            "ancestor_mask",
            &contract.target,
        )?;
        let target_position_input = role(
            &contract.target_port_bindings,
            "position_ids",
            &contract.target,
        )?;
        let target_accepted_input = role(
            &contract.target_port_bindings,
            "accepted_tokens",
            &contract.target,
        )?;
        let output_value = |component: &str,
                            output: &str,
                            outputs: &std::collections::BTreeMap<String, String>|
         -> Result<String, String> {
            outputs.get(output).cloned().ok_or_else(|| {
                format!("candidate-tree output '{component}::{output}' has no workflow SSA binding")
            })
        };
        let candidate_value = output_value(&contract.proposer, candidate_tokens, proposer_outputs)?;
        let actual_candidate = target_bindings
            .get(&target_candidate_input)
            .ok_or_else(|| {
                format!("{target_path}.inputs.{target_candidate_input} has no workflow SSA binding")
            })?
            .clone();
        require_exact_candidate_value(
            workflow,
            &workflow.steps,
            ExactBindingExpectation {
                target_path: &target_path,
                target_port: &target_candidate_input,
                producer: &contract.proposer,
                producer_port: candidate_tokens,
                expected: &candidate_value,
            },
            &actual_candidate,
        )?;

        let topology_output = match topology {
            CandidateTreeTopology::ParentIndices { output }
            | CandidateTreeTopology::AncestorMask { output } => output,
        };
        let topology_value = output_value(&contract.proposer, topology_output, proposer_outputs)?;
        let actual_topology = target_bindings
            .get(&target_topology_input)
            .ok_or_else(|| {
                format!("{target_path}.inputs.{target_topology_input} has no workflow SSA binding")
            })?
            .clone();
        let target_topology_value = match topology {
            CandidateTreeTopology::AncestorMask { .. } => {
                require_exact_candidate_value(
                    workflow,
                    &workflow.steps,
                    ExactBindingExpectation {
                        target_path: &target_path,
                        target_port: &target_topology_input,
                        producer: &contract.proposer,
                        producer_port: topology_output,
                        expected: &topology_value,
                    },
                    &actual_topology,
                )?;
                CandidateTreeTopologyInput::ProposerValue {
                    value: topology_value,
                }
            }
            CandidateTreeTopology::ParentIndices { .. } => {
                require_driver_placeholder(
                    workflow,
                    &target_path,
                    &target_topology_input,
                    &actual_topology,
                    "ancestor mask derived from the proved parent-index topology",
                )?;
                CandidateTreeTopologyInput::DerivedFromParentIndices {
                    topology_value,
                    placeholder: actual_topology,
                }
            }
        };
        let target_position_placeholder = target_bindings
            .get(&target_position_input)
            .ok_or_else(|| {
                format!("{target_path}.inputs.{target_position_input} has no workflow SSA binding")
            })?
            .clone();
        require_driver_placeholder(
            workflow,
            &target_path,
            &target_position_input,
            &target_position_placeholder,
            "position IDs derived from the proved candidate topology",
        )?;
        let target_accepted_placeholder = target_bindings
            .get(&target_accepted_input)
            .ok_or_else(|| {
                format!("{target_path}.inputs.{target_accepted_input} has no workflow SSA binding")
            })?
            .clone();
        require_driver_placeholder(
            workflow,
            &target_path,
            &target_accepted_input,
            &target_accepted_placeholder,
            "accepted-token context produced by runtime verification",
        )?;
        if workflow.inputs.contains_key(binding) {
            require_driver_placeholder(
                workflow,
                &target_path,
                "speculative.verification.accepted_path",
                binding,
                "the runtime-selected accepted candidate tokens",
            )?;
        } else if let Some((component, port)) =
            component_output_provenance(&workflow.steps, binding)
        {
            return Err(format!(
                "speculative.verification.accepted_path.binding '{binding}' collides with \
                 component '{component}' output port '{port}'; use an unowned runtime binding or \
                 an optional opaque literal placeholder so accepted-path publication has one \
                 authority"
            ));
        }
        let target_logits_value = output_value(
            &contract.verification.target_output.component,
            &contract.verification.target_output.output,
            target_outputs,
        )?;
        let (proposal_probabilities_value, target_probabilities_value) =
            match &contract.verification.probabilities {
                Some(probabilities) => (
                    Some(output_value(
                        &probabilities.proposal.component,
                        &probabilities.proposal.output,
                        proposer_outputs,
                    )?),
                    Some(output_value(
                        &probabilities.target.component,
                        &probabilities.target.output,
                        target_outputs,
                    )?),
                ),
                None => (None, None),
            };
        let (proposer_location, target_location, execution_mode) =
            candidate_tree_seam_locations(workflow, &contract.proposer, &contract.target)?;
        let token_outputs = workflow
            .outputs
            .iter()
            .filter_map(|(name, output)| {
                (output.role == WorkflowOutputRole::Tokens).then_some(name.clone())
            })
            .collect::<Vec<_>>();
        let [token_output] = token_outputs.as_slice() else {
            return Err(format!(
                "candidate-tree generation requires exactly one workflow output with role \
                 'tokens', but the authored workflow declares {token_outputs:?}"
            ));
        };
        let token_output_authority = resolve_candidate_tree_token_output_authority(
            workflow,
            binding,
            token_output,
            &target_location,
        )?;
        let plan = Self {
            version: contract.version.clone(),
            proposer: contract.proposer.clone(),
            target: contract.target.clone(),
            proposer_path: proposer_location.path,
            target_path: target_location.path,
            control_provenance: proposer_location.control,
            execution_mode,
            topology: topology.clone(),
            accepted_path_binding: binding.clone(),
            proposer_bindings: proposer_bindings.clone(),
            proposer_outputs: proposer_outputs.clone(),
            target_bindings: target_bindings.clone(),
            target_outputs: target_outputs.clone(),
            proposer_context_input,
            target_context_input,
            target_candidate_input,
            target_candidate_value: candidate_value,
            target_topology_input,
            target_topology_value,
            target_position_input,
            target_position_placeholder,
            target_accepted_input,
            target_accepted_placeholder,
            target_logits_value,
            proposal_probabilities_value,
            target_probabilities_value,
            rollback_state: contract.rollback_state.clone(),
            max_proposal_width: contract.max_proposal_width,
            token_output_authority,
        };
        plan.validate_state_authority(workflow)?;
        Ok(plan)
    }

    fn validate_state_authority(&self, workflow: &WorkflowSpec) -> Result<(), String> {
        let state_plan = onnx_genai_metadata::resolve_state_plan(workflow);
        for cell in &self.rollback_state {
            let resolved = state_plan.cell(cell).cloned().ok_or_else(|| {
                format!("candidate-tree rollback participant '{cell}' is unresolved")
            })?;
            resolved.final_writer.ok_or_else(|| {
                format!(
                    "candidate-tree rollback participant '{cell}' has no canonical final writer"
                )
            })?;
        }
        if let Some(serving) = &workflow.serving {
            for group in serving.state_service.groups.values() {
                for component in [&self.proposer, &self.target] {
                    for (cell, alias) in group.ports.get(component).into_iter().flatten() {
                        if alias.access == StatePortAccess::ReadWrite
                            && !self.rollback_state.contains(cell)
                        {
                            return Err(format!(
                                "candidate-tree component '{component}' mutates state '{cell}', \
                                 but speculative.rollback_state omits it"
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub(crate) fn state_alias<'a>(
        &self,
        workflow: &'a WorkflowSpec,
        component: &str,
        cell: &str,
    ) -> Option<&'a onnx_genai_metadata::StatePortAlias> {
        workflow
            .serving
            .as_ref()?
            .state_service
            .groups
            .values()
            .find_map(|group| group.ports.get(component)?.get(cell))
            .filter(|alias| alias.access == StatePortAccess::ReadWrite)
    }
}

struct ExactBindingExpectation<'a> {
    target_path: &'a str,
    target_port: &'a str,
    producer: &'a str,
    producer_port: &'a str,
    expected: &'a str,
}

fn require_exact_candidate_value(
    workflow: &WorkflowSpec,
    steps: &[WorkflowStep],
    expectation: ExactBindingExpectation<'_>,
    actual: &str,
) -> Result<(), String> {
    if actual == expectation.expected {
        return Ok(());
    }
    Err(format!(
        "{}.inputs.{} must consume proposer '{}' output port '{}' as exact SSA value '{}', but it \
         binds '{actual}', whose \
         provenance is {}. Connect the target input directly to '{expected}'; dtype, shape, \
         port spelling, request values, transformed values, and unrelated component outputs do \
         not prove candidate identity",
        expectation.target_path,
        expectation.target_port,
        expectation.producer,
        expectation.producer_port,
        expectation.expected,
        describe_value_provenance(workflow, steps, actual),
        expected = expectation.expected,
    ))
}

fn require_driver_placeholder(
    workflow: &WorkflowSpec,
    target_path: &str,
    target_port: &str,
    binding: &str,
    derived: &str,
) -> Result<(), String> {
    let Some(input) = workflow.inputs.get(binding) else {
        return Err(format!(
            "{target_path}.inputs.{target_port} binds '{binding}', whose provenance is not a \
             workflow input. The candidate-tree driver supplies {derived}; bind an optional \
             opaque literal placeholder so no request or component value is silently overridden"
        ));
    };
    if !matches!(
        input.source,
        onnx_genai_metadata::WorkflowInputSource::Literal
    ) || input.required
        || input.default.is_none()
        || !matches!(input.role, SemanticInputRole::Opaque)
    {
        return Err(format!(
            "{target_path}.inputs.{target_port} binds '{binding}' ({:?}, required={}, default={}), \
             but the candidate-tree driver supplies {derived}. Use an optional opaque literal \
             placeholder; request/application inputs and component outputs cannot be ignored",
            input.source,
            input.required,
            input.default.is_some()
        ));
    }
    Ok(())
}

fn describe_value_provenance(
    workflow: &WorkflowSpec,
    steps: &[WorkflowStep],
    value: &str,
) -> String {
    if let Some(input) = workflow.inputs.get(value) {
        return format!(
            "workflow input '{value}' with source {:?} and role {:?}",
            input.source, input.role
        );
    }
    if let Some((component, port)) = component_output_provenance(steps, value) {
        return format!("component '{component}' output port '{port}'");
    }
    format!("SSA value '{value}' with no unique proved source")
}

fn component_output_provenance<'a>(
    steps: &'a [WorkflowStep],
    value: &str,
) -> Option<(&'a str, &'a str)> {
    for step in steps {
        match step {
            WorkflowStep::Invoke {
                component, outputs, ..
            } => {
                if let Some((port, _)) = outputs.iter().find(|(_, output)| output.as_str() == value)
                {
                    return Some((component, port));
                }
            }
            WorkflowStep::Sequence { steps } => {
                if let Some(source) = component_output_provenance(steps, value) {
                    return Some(source);
                }
            }
            WorkflowStep::Loop { setup, steps, .. } => {
                if let Some(source) = component_output_provenance(setup, value)
                    .or_else(|| component_output_provenance(steps, value))
                {
                    return Some(source);
                }
            }
            WorkflowStep::Branch { cases, default, .. } => {
                for case in cases.values() {
                    if let Some(source) =
                        component_output_provenance(std::slice::from_ref(case), value)
                    {
                        return Some(source);
                    }
                }
                if let Some(default) = default
                    && let Some(source) =
                        component_output_provenance(std::slice::from_ref(default), value)
                {
                    return Some(source);
                }
            }
            WorkflowStep::Emit { .. } => {}
        }
    }
    None
}

fn candidate_tree_unavailable_reason(
    contract: &SpeculativeContract,
    workflow: Option<&onnx_genai_metadata::WorkflowSpec>,
) -> Option<String> {
    if contract.version != "1" {
        return Some(format!(
            "contract version '{}' is unsupported; this runtime implements exactly version '1'",
            contract.version
        ));
    }
    if !matches!(
        contract.verification.accepted_path,
        SpeculativeAcceptedPath::Runtime { .. }
    ) {
        return Some(
            "component-owned accepted paths are unsupported; version-1 execution requires the \
                     declared runtime accepted-prefix binding"
                .to_string(),
        );
    }
    let Some(workflow) = workflow else {
        return Some("the package has no pipeline.workflow execution graph".to_string());
    };
    if !workflow
        .outputs
        .values()
        .any(|output| output.role == WorkflowOutputRole::Tokens)
    {
        return Some(
            "the workflow does not declare an output with role 'tokens' required for \
             candidate-tree generation"
                .to_string(),
        );
    }
    for component in [&contract.proposer, &contract.target] {
        let Some(declaration) = workflow.components.get(component) else {
            return Some(format!("component '{component}' is undeclared"));
        };
        if !matches!(
            declaration.implementation,
            ComponentImplementation::Onnx { .. }
        ) {
            return Some(format!(
                "component '{component}' is not an ONNX artifact; binding/adapter candidate-tree \
                         components have no production execution authority"
            ));
        }
        if invocation_count(&workflow.steps, component) != 1 {
            return Some(format!(
                "component '{component}' must have exactly one workflow invocation whose typed \
                         bindings define the candidate-tree ABI"
            ));
        }
    }
    for (role, ports, component) in [
        (
            "context_tokens",
            &contract.port_bindings,
            contract.proposer.as_str(),
        ),
        (
            "context_tokens",
            &contract.target_port_bindings,
            contract.target.as_str(),
        ),
        (
            "candidate_tokens",
            &contract.target_port_bindings,
            contract.target.as_str(),
        ),
        (
            "ancestor_mask",
            &contract.target_port_bindings,
            contract.target.as_str(),
        ),
        (
            "position_ids",
            &contract.target_port_bindings,
            contract.target.as_str(),
        ),
        (
            "accepted_tokens",
            &contract.target_port_bindings,
            contract.target.as_str(),
        ),
    ] {
        let Some(port) = ports.get(role) else {
            return Some(format!(
                "candidate-tree role '{role}' is absent for component '{component}'"
            ));
        };
        let Some(declaration) = workflow.components.get(component) else {
            continue;
        };
        if !declaration.ports.inputs.contains_key(port) {
            return Some(format!(
                "candidate-tree role '{role}' selects absent input port '{port}' on component \
                         '{component}'"
            ));
        }
    }
    let Some((proposer_inputs, proposer_outputs)) =
        component_invocation(&workflow.steps, &contract.proposer)
    else {
        return Some("the proposer invocation disappeared during admission".to_string());
    };
    let Some((target_inputs, target_outputs)) =
        component_invocation(&workflow.steps, &contract.target)
    else {
        return Some("the target invocation disappeared during admission".to_string());
    };
    for (role, port) in &contract.port_bindings {
        if !proposer_inputs.contains_key(port) {
            return Some(format!(
                "candidate-tree proposer role '{role}' selects input '{port}', but its workflow \
                 invocation does not bind that port"
            ));
        }
    }
    for (role, port) in &contract.target_port_bindings {
        if !target_inputs.contains_key(port) {
            return Some(format!(
                "candidate-tree target role '{role}' selects input '{port}', but its workflow \
                 invocation does not bind that port"
            ));
        }
    }
    let SpeculativeProposalExecution::CandidateTree {
        candidate_tokens,
        topology,
    } = &contract.proposal_execution
    else {
        unreachable!("candidate-tree admission is called only for candidate-tree contracts");
    };
    let topology_output = match topology {
        onnx_genai_metadata::CandidateTreeTopology::ParentIndices { output }
        | onnx_genai_metadata::CandidateTreeTopology::AncestorMask { output } => output,
    };
    for output in [candidate_tokens, topology_output] {
        if !proposer_outputs.contains_key(output) {
            return Some(format!(
                "candidate-tree proposer output '{output}' is not bound by its workflow invocation"
            ));
        }
    }
    if !target_outputs.contains_key(&contract.verification.target_output.output) {
        return Some(format!(
            "candidate-tree target verifier output '{}' is not bound by its workflow invocation",
            contract.verification.target_output.output
        ));
    }
    if let Some(probabilities) = &contract.verification.probabilities {
        if !proposer_outputs.contains_key(&probabilities.proposal.output) {
            return Some(format!(
                "candidate-tree proposal probability output '{}' is not bound by its workflow \
                 invocation",
                probabilities.proposal.output
            ));
        }
        if !target_outputs.contains_key(&probabilities.target.output) {
            return Some(format!(
                "candidate-tree target probability output '{}' is not bound by its workflow \
                 invocation",
                probabilities.target.output
            ));
        }
    }
    None
}

fn invocation_count(steps: &[WorkflowStep], component: &str) -> usize {
    steps
        .iter()
        .map(|step| match step {
            WorkflowStep::Invoke {
                component: invoked, ..
            } => usize::from(invoked == component),
            WorkflowStep::Sequence { steps } => invocation_count(steps, component),
            WorkflowStep::Loop { setup, steps, .. } => {
                invocation_count(setup, component) + invocation_count(steps, component)
            }
            WorkflowStep::Branch { cases, default, .. } => {
                cases
                    .values()
                    .map(|step| invocation_count(std::slice::from_ref(step), component))
                    .sum::<usize>()
                    + default
                        .as_deref()
                        .map(|step| invocation_count(std::slice::from_ref(step), component))
                        .unwrap_or_default()
            }
            WorkflowStep::Emit { .. } => 0,
        })
        .sum()
}

fn component_invocation<'a>(
    steps: &'a [WorkflowStep],
    component: &str,
) -> Option<InvocationBindings<'a>> {
    for step in steps {
        match step {
            WorkflowStep::Invoke {
                component: invoked,
                inputs,
                outputs,
            } if invoked == component => return Some((inputs, outputs)),
            WorkflowStep::Sequence { steps } => {
                if let Some(found) = component_invocation(steps, component) {
                    return Some(found);
                }
            }
            WorkflowStep::Loop { setup, steps, .. } => {
                if let Some(found) = component_invocation(setup, component)
                    .or_else(|| component_invocation(steps, component))
                {
                    return Some(found);
                }
            }
            WorkflowStep::Branch { cases, default, .. } => {
                for case in cases.values() {
                    if let Some(found) = component_invocation(std::slice::from_ref(case), component)
                    {
                        return Some(found);
                    }
                }
                if let Some(default) = default
                    && let Some(found) =
                        component_invocation(std::slice::from_ref(default), component)
                {
                    return Some(found);
                }
            }
            _ => {}
        }
    }
    None
}

fn component_invocation_path(
    steps: &[WorkflowStep],
    component: &str,
    path: &str,
) -> Option<String> {
    for (index, step) in steps.iter().enumerate() {
        let step_path = format!("{path}[{index}]");
        match step {
            WorkflowStep::Invoke {
                component: invoked, ..
            } if invoked == component => return Some(step_path),
            WorkflowStep::Sequence { steps } => {
                if let Some(found) =
                    component_invocation_path(steps, component, &format!("{step_path}.steps"))
                {
                    return Some(found);
                }
            }
            WorkflowStep::Loop { setup, steps, .. } => {
                if let Some(found) =
                    component_invocation_path(setup, component, &format!("{step_path}.setup"))
                        .or_else(|| {
                            component_invocation_path(
                                steps,
                                component,
                                &format!("{step_path}.steps"),
                            )
                        })
                {
                    return Some(found);
                }
            }
            WorkflowStep::Branch { cases, default, .. } => {
                for (case, branch) in cases {
                    if let Some(found) = component_invocation_path(
                        std::slice::from_ref(branch),
                        component,
                        &format!("{step_path}.cases[{case}]"),
                    ) {
                        return Some(found);
                    }
                }
                if let Some(default) = default
                    && let Some(found) = component_invocation_path(
                        std::slice::from_ref(default),
                        component,
                        &format!("{step_path}.default"),
                    )
                {
                    return Some(found);
                }
            }
            _ => {}
        }
    }
    None
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CandidateTreeInvocationLocation {
    path: String,
    control: Vec<CandidateTreeControlFrame>,
}

#[derive(Debug)]
enum CandidateTreeRegionStep {
    Invoke {
        component: String,
        location: CandidateTreeInvocationLocation,
    },
    Boundary {
        path: String,
        kind: &'static str,
    },
    Emit {
        path: String,
        output: String,
        value: String,
        when: Option<String>,
        valid_length: Option<String>,
        mode: onnx_genai_metadata::WorkflowEmitMode,
    },
}

fn candidate_tree_seam_locations(
    workflow: &WorkflowSpec,
    proposer: &str,
    target: &str,
) -> Result<
    (
        CandidateTreeInvocationLocation,
        CandidateTreeInvocationLocation,
        CandidateTreeExecutionMode,
    ),
    String,
> {
    let mut regions = std::collections::BTreeMap::new();
    collect_candidate_tree_regions(
        &workflow.steps,
        "pipeline.workflow.steps",
        &[],
        &mut regions,
    );
    let mut proposer_location = None;
    let mut target_location = None;
    let mut proposer_position = None;
    let mut target_position = None;
    for (control, steps) in &regions {
        for (index, step) in steps.iter().enumerate() {
            let CandidateTreeRegionStep::Invoke {
                component,
                location,
            } = step
            else {
                continue;
            };
            if component == proposer {
                proposer_location = Some(location.clone());
                proposer_position = Some((control.clone(), index));
            }
            if component == target {
                target_location = Some(location.clone());
                target_position = Some((control.clone(), index));
            }
        }
    }
    let proposer_location = proposer_location
        .ok_or_else(|| format!("candidate-tree proposer '{proposer}' has no workflow location"))?;
    let target_location = target_location
        .ok_or_else(|| format!("candidate-tree target '{target}' has no workflow location"))?;
    let (proposer_control, proposer_index) =
        proposer_position.expect("candidate-tree proposer location has a region position");
    let (target_control, target_index) =
        target_position.expect("candidate-tree target location has a region position");
    if proposer_control != target_control {
        return Err(format!(
            "candidate-tree proposer '{}' at {} has control provenance {:?}, but target '{}' at \
             {} has {:?}; one semantic execution cannot prove that both run on the same branch \
             and loop entry. Place the exact proposer→target verification seam in one sequence \
             region and join branch-local inputs through declared phi values before it",
            proposer,
            proposer_location.path,
            proposer_location.control,
            target,
            target_location.path,
            target_location.control
        ));
    }
    let steps = regions
        .get(&proposer_control)
        .expect("candidate-tree control region was collected");
    if target_index != proposer_index + 1 {
        let detail = steps
            .get(proposer_index + 1)
            .map(|step| match step {
                CandidateTreeRegionStep::Invoke {
                    component,
                    location,
                } => format!("invocation '{}' at {}", component, location.path),
                CandidateTreeRegionStep::Boundary { path, kind } => {
                    format!("{kind} at {path}")
                }
                CandidateTreeRegionStep::Emit { path, output, .. } => {
                    format!("emit into '{output}' at {path}")
                }
            })
            .unwrap_or_else(|| "the end of the region".to_string());
        return Err(format!(
            "candidate-tree proposer '{}' at {} is not immediately followed by target '{}' at \
             {} in their common control region; {detail} lies at the verification seam. Move \
             unrelated work before or after the exact proposer→target pair so speculative replay \
             cannot skip or duplicate it",
            proposer, proposer_location.path, target, target_location.path
        ));
    }
    let execution_mode = if proposer_control
        .iter()
        .any(|frame| matches!(frame, CandidateTreeControlFrame::LoopBody { .. }))
    {
        CandidateTreeExecutionMode::OncePerAuthoredEntry
    } else {
        CandidateTreeExecutionMode::DrainAtSeam
    };
    Ok((proposer_location, target_location, execution_mode))
}

fn collect_candidate_tree_regions(
    steps: &[WorkflowStep],
    path: &str,
    control: &[CandidateTreeControlFrame],
    regions: &mut std::collections::BTreeMap<
        Vec<CandidateTreeControlFrame>,
        Vec<CandidateTreeRegionStep>,
    >,
) {
    for (index, step) in steps.iter().enumerate() {
        let step_path = format!("{path}[{index}]");
        match step {
            WorkflowStep::Invoke { component, .. } => {
                regions.entry(control.to_vec()).or_default().push(
                    CandidateTreeRegionStep::Invoke {
                        component: component.clone(),
                        location: CandidateTreeInvocationLocation {
                            path: step_path,
                            control: control.to_vec(),
                        },
                    },
                );
            }
            WorkflowStep::Sequence { steps } => collect_candidate_tree_regions(
                steps,
                &format!("{step_path}.steps"),
                control,
                regions,
            ),
            WorkflowStep::Loop { setup, steps, .. } => {
                regions.entry(control.to_vec()).or_default().push(
                    CandidateTreeRegionStep::Boundary {
                        path: step_path.clone(),
                        kind: "loop",
                    },
                );
                let mut setup_control = control.to_vec();
                setup_control.push(CandidateTreeControlFrame::LoopSetup {
                    path: step_path.clone(),
                });
                collect_candidate_tree_regions(
                    setup,
                    &format!("{step_path}.setup"),
                    &setup_control,
                    regions,
                );
                let mut body_control = control.to_vec();
                body_control.push(CandidateTreeControlFrame::LoopBody {
                    path: step_path.clone(),
                });
                collect_candidate_tree_regions(
                    steps,
                    &format!("{step_path}.steps"),
                    &body_control,
                    regions,
                );
            }
            WorkflowStep::Branch { cases, default, .. } => {
                regions.entry(control.to_vec()).or_default().push(
                    CandidateTreeRegionStep::Boundary {
                        path: step_path.clone(),
                        kind: "branch",
                    },
                );
                for (key, branch) in cases {
                    let mut branch_control = control.to_vec();
                    branch_control.push(CandidateTreeControlFrame::BranchCase {
                        path: step_path.clone(),
                        key: key.clone(),
                    });
                    collect_candidate_tree_regions(
                        std::slice::from_ref(branch),
                        &format!("{step_path}.cases[{key}]"),
                        &branch_control,
                        regions,
                    );
                }
                if let Some(default) = default {
                    let mut default_control = control.to_vec();
                    default_control.push(CandidateTreeControlFrame::BranchDefault {
                        path: step_path.clone(),
                    });
                    collect_candidate_tree_regions(
                        std::slice::from_ref(default),
                        &format!("{step_path}.default"),
                        &default_control,
                        regions,
                    );
                }
            }
            WorkflowStep::Emit {
                output,
                value,
                when,
                valid_length,
                mode,
                ..
            } => {
                regions
                    .entry(control.to_vec())
                    .or_default()
                    .push(CandidateTreeRegionStep::Emit {
                        path: step_path,
                        output: output.clone(),
                        value: value.clone(),
                        when: when.clone(),
                        valid_length: valid_length.clone(),
                        mode: mode.clone(),
                    });
            }
        }
    }
}

#[derive(Debug, Clone)]
struct CandidateTreeTokenEmitDefinition {
    path: String,
    control: Vec<CandidateTreeControlFrame>,
    region_index: usize,
    value: String,
    when: Option<String>,
    valid_length: Option<String>,
    mode: onnx_genai_metadata::WorkflowEmitMode,
}

#[derive(Debug, Clone)]
struct CandidateTreePhiDefinition {
    path: String,
    control: Vec<CandidateTreeControlFrame>,
    region_index: usize,
    incoming: Vec<(String, String)>,
}

fn resolve_candidate_tree_token_output_authority(
    workflow: &WorkflowSpec,
    accepted_path_binding: &str,
    token_output: &str,
    target_location: &CandidateTreeInvocationLocation,
) -> Result<CandidateTreeTokenOutputAuthority, String> {
    let mut regions = std::collections::BTreeMap::new();
    collect_candidate_tree_regions(
        &workflow.steps,
        "pipeline.workflow.steps",
        &[],
        &mut regions,
    );
    let (target_control, target_index) = regions
        .iter()
        .find_map(|(control, steps)| {
            steps.iter().enumerate().find_map(|(index, step)| {
                matches!(
                    step,
                    CandidateTreeRegionStep::Invoke { location, .. }
                        if location.path == target_location.path
                )
                .then(|| (control.clone(), index))
            })
        })
        .ok_or_else(|| {
            format!(
                "candidate-tree target at {} disappeared while proving the canonical generated-token output",
                target_location.path
            )
        })?;

    let mut branch_regions = std::collections::BTreeMap::new();
    let mut token_emits = Vec::new();
    for (control, steps) in &regions {
        for (region_index, step) in steps.iter().enumerate() {
            match step {
                CandidateTreeRegionStep::Boundary {
                    path,
                    kind: "branch",
                } => {
                    branch_regions.insert(path.clone(), (control.clone(), region_index));
                }
                CandidateTreeRegionStep::Emit {
                    path,
                    output,
                    value,
                    when,
                    valid_length,
                    mode,
                } if output == token_output => {
                    token_emits.push(CandidateTreeTokenEmitDefinition {
                        path: path.clone(),
                        control: control.clone(),
                        region_index,
                        value: value.clone(),
                        when: when.clone(),
                        valid_length: valid_length.clone(),
                        mode: mode.clone(),
                    });
                }
                _ => {}
            }
        }
    }

    if token_emits.len() != 1 {
        let paths = token_emits
            .iter()
            .map(|emit| emit.path.as_str())
            .collect::<Vec<_>>();
        return Err(format!(
            "candidate-tree canonical generated-token output '{token_output}' requires exactly one \
             authored emit site, but found {} at {paths:?}; host publication is unavailable, so \
             remove duplicates and emit the accepted-path binding once",
            token_emits.len()
        ));
    }
    let emit = token_emits
        .pop()
        .expect("the exactly-one canonical token emit is present");
    if emit.control != target_control || emit.region_index <= target_index {
        return Err(format!(
            "candidate-tree canonical generated-token emit at {} is not dominated by target {} \
             on the same control path. Setup, root, branch-local, after-loop, and zero-trip \
             paths cannot publish the accepted path; place the emit after the target in its \
             exact sequence region",
            emit.path, target_location.path
        ));
    }
    if emit.when.is_some() || emit.valid_length.is_some() {
        return Err(format!(
            "candidate-tree canonical generated-token emit at {} conditionally suppresses or \
             slices the accepted path (when={:?}, valid_length={:?}); publish the complete \
             committed accepted-path binding without a guard or transform",
            emit.path, emit.when, emit.valid_length
        ));
    }
    if !matches!(
        emit.mode,
        onnx_genai_metadata::WorkflowEmitMode::Append
            | onnx_genai_metadata::WorkflowEmitMode::Event
    ) {
        return Err(format!(
            "candidate-tree canonical generated-token emit at {} uses {:?}; use append or event \
             so every committed accepted-path record remains visible to S4 and GenerateResult",
            emit.path, emit.mode
        ));
    }

    let mut phi_definitions = std::collections::BTreeMap::new();
    collect_candidate_tree_phi_definitions(
        &workflow.steps,
        "pipeline.workflow.steps",
        &[],
        &branch_regions,
        &mut phi_definitions,
    )?;
    prove_candidate_tree_token_output_value(
        workflow,
        &emit.value,
        accepted_path_binding,
        &target_control,
        target_index,
        &phi_definitions,
        &mut std::collections::BTreeSet::new(),
    )
    .map_err(|reason| {
        format!(
            "candidate-tree canonical generated-token emit at {} does not consume accepted-path \
             binding '{accepted_path_binding}': {reason}",
            emit.path
        )
    })?;

    Ok(CandidateTreeTokenOutputAuthority::AuthoredEmit {
        output: token_output.to_string(),
        sites: vec![CandidateTreeTokenEmitSite { path: emit.path }],
    })
}

fn collect_candidate_tree_phi_definitions(
    steps: &[WorkflowStep],
    path: &str,
    control: &[CandidateTreeControlFrame],
    branch_regions: &std::collections::BTreeMap<String, (Vec<CandidateTreeControlFrame>, usize)>,
    definitions: &mut std::collections::BTreeMap<String, Vec<CandidateTreePhiDefinition>>,
) -> Result<(), String> {
    for (index, step) in steps.iter().enumerate() {
        let step_path = format!("{path}[{index}]");
        match step {
            WorkflowStep::Sequence { steps } => collect_candidate_tree_phi_definitions(
                steps,
                &format!("{step_path}.steps"),
                control,
                branch_regions,
                definitions,
            )?,
            WorkflowStep::Loop { setup, steps, .. } => {
                let mut setup_control = control.to_vec();
                setup_control.push(CandidateTreeControlFrame::LoopSetup {
                    path: step_path.clone(),
                });
                collect_candidate_tree_phi_definitions(
                    setup,
                    &format!("{step_path}.setup"),
                    &setup_control,
                    branch_regions,
                    definitions,
                )?;
                let mut body_control = control.to_vec();
                body_control.push(CandidateTreeControlFrame::LoopBody {
                    path: step_path.clone(),
                });
                collect_candidate_tree_phi_definitions(
                    steps,
                    &format!("{step_path}.steps"),
                    &body_control,
                    branch_regions,
                    definitions,
                )?;
            }
            WorkflowStep::Branch {
                cases,
                default,
                outputs,
                ..
            } => {
                let (definition_control, region_index) =
                    branch_regions.get(&step_path).cloned().ok_or_else(|| {
                        format!(
                            "branch at {step_path} is absent from candidate-tree dominance analysis"
                        )
                    })?;
                for (value, phi) in outputs {
                    let mut incoming =
                        Vec::with_capacity(cases.len() + usize::from(default.is_some()));
                    for case in cases.keys() {
                        let source = phi.cases.get(case).ok_or_else(|| {
                            format!(
                                "transparent phi '{value}' at {step_path}.outputs.{value} has no \
                                 input for reachable branch case '{case}'"
                            )
                        })?;
                        incoming.push((format!("case '{case}'"), source.clone()));
                    }
                    if default.is_some() {
                        let source = phi.default.as_ref().ok_or_else(|| {
                            format!(
                                "transparent phi '{value}' at {step_path}.outputs.{value} has no \
                                 input for its reachable default branch"
                            )
                        })?;
                        incoming.push(("default".to_string(), source.clone()));
                    }
                    definitions.entry(value.clone()).or_default().push(
                        CandidateTreePhiDefinition {
                            path: format!("{step_path}.outputs.{value}"),
                            control: definition_control.clone(),
                            region_index,
                            incoming,
                        },
                    );
                }
                for (key, branch) in cases {
                    let mut branch_control = control.to_vec();
                    branch_control.push(CandidateTreeControlFrame::BranchCase {
                        path: step_path.clone(),
                        key: key.clone(),
                    });
                    collect_candidate_tree_phi_definitions(
                        std::slice::from_ref(branch),
                        &format!("{step_path}.cases[{key}]"),
                        &branch_control,
                        branch_regions,
                        definitions,
                    )?;
                }
                if let Some(default) = default {
                    let mut branch_control = control.to_vec();
                    branch_control.push(CandidateTreeControlFrame::BranchDefault {
                        path: step_path.clone(),
                    });
                    collect_candidate_tree_phi_definitions(
                        std::slice::from_ref(default),
                        &format!("{step_path}.default"),
                        &branch_control,
                        branch_regions,
                        definitions,
                    )?;
                }
            }
            WorkflowStep::Invoke { .. } | WorkflowStep::Emit { .. } => {}
        }
    }
    Ok(())
}

fn prove_candidate_tree_token_output_value(
    workflow: &WorkflowSpec,
    value: &str,
    accepted_path_binding: &str,
    target_control: &[CandidateTreeControlFrame],
    target_index: usize,
    phi_definitions: &std::collections::BTreeMap<String, Vec<CandidateTreePhiDefinition>>,
    visiting: &mut std::collections::BTreeSet<String>,
) -> Result<(), String> {
    if value == accepted_path_binding {
        return Ok(());
    }
    let definitions = phi_definitions.get(value).ok_or_else(|| {
        format!(
            "value '{value}' is not the accepted-path binding and has no declared transparent \
             phi provenance; its provenance is {}",
            describe_value_provenance(workflow, &workflow.steps, value)
        )
    })?;
    let [definition] = definitions.as_slice() else {
        return Err(format!(
            "value '{value}' has {} transparent phi definitions, so its accepted-path identity \
             is ambiguous",
            definitions.len()
        ));
    };
    if definition.control != target_control || definition.region_index <= target_index {
        return Err(format!(
            "transparent phi '{}' at {} does not execute after the candidate target on its exact \
             control path, so it may preserve an earlier placeholder rather than the committed \
             accepted path",
            value, definition.path
        ));
    }
    if !visiting.insert(value.to_string()) {
        return Err(format!(
            "transparent phi '{}' at {} is cyclic",
            value, definition.path
        ));
    }
    let result = definition.incoming.iter().try_for_each(|(arm, source)| {
        prove_candidate_tree_token_output_value(
            workflow,
            source,
            accepted_path_binding,
            target_control,
            target_index,
            phi_definitions,
            visiting,
        )
        .map_err(|reason| {
            format!(
                "{arm} of transparent phi '{}' at {}: {reason}",
                value, definition.path
            )
        })
    });
    visiting.remove(value);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_and_canonical_chained_mtp_remain_admitted() {
        assert_eq!(
            WorkflowExecutionAdmission::from_metadata(
                &InferenceMetadata::default(),
                EngineDecodeBackend::Ort,
            ),
            WorkflowExecutionAdmission::Admitted
        );
        let mtp = onnx_genai_metadata::parse_metadata(
                include_str!(
                    "../../../../examples/inference_metadata/catalogue/22-qwen3-chained-speculative-decoding.yaml"
                ),
                Some("yaml"),
            )
            .expect("canonical chained MTP fixture parses");
        assert_eq!(
            WorkflowExecutionAdmission::from_metadata(&mtp, EngineDecodeBackend::Ort),
            WorkflowExecutionAdmission::Admitted
        );
    }

    #[test]
    fn only_the_implemented_dflash_v1_ort_pair_is_admitted() {
        let dflash = onnx_genai_metadata::parse_metadata(
            include_str!("../../tests/fixtures/dflash-admission/inference_metadata.yaml"),
            Some("yaml"),
        )
        .expect("DFlash fixture parses");
        assert_eq!(
            WorkflowExecutionAdmission::from_metadata(&dflash, EngineDecodeBackend::Ort),
            WorkflowExecutionAdmission::Admitted
        );
        assert_eq!(
            WorkflowExecutionAdmission::from_metadata(&dflash, EngineDecodeBackend::Native),
            WorkflowExecutionAdmission::DFlashUnavailable {
                version: "1".to_string(),
            }
        );

        let mut versioned = dflash;
        let SpeculativeProposalExecution::DflashFlatBlock {
            version: declared, ..
        } = &mut versioned
            .speculative
            .as_mut()
            .expect("fixture declares speculation")
            .proposal_execution
        else {
            panic!("fixture declares DFlash")
        };
        *declared = "2".to_string();
        assert_eq!(
            WorkflowExecutionAdmission::from_metadata(&versioned, EngineDecodeBackend::Ort),
            WorkflowExecutionAdmission::DFlashUnavailable {
                version: "2".to_string(),
            }
        );
    }

    #[test]
    fn candidate_tree_and_dflash_are_independent_exact_capabilities() {
        let candidate = onnx_genai_metadata::parse_metadata(
            include_str!("../../tests/fixtures/unsupported-candidate-tree/inference_metadata.yaml"),
            Some("yaml"),
        )
        .expect("candidate-tree fixture parses");
        assert!(matches!(
            WorkflowExecutionAdmission::from_metadata(&candidate, EngineDecodeBackend::Ort),
            WorkflowExecutionAdmission::CandidateTreeUnavailable {
                version,
                reason,
            }
                if version == "1" && !reason.is_empty()
        ));
    }

    #[test]
    fn canonical_candidate_tree_fixture_is_admitted() {
        let tree = onnx_genai_metadata::parse_metadata(
            include_str!("../../tests/fixtures/unsupported-candidate-tree/inference_metadata.yaml"),
            Some("yaml"),
        )
        .expect("candidate tree parses");
        assert!(matches!(
            WorkflowExecutionAdmission::from_metadata(&tree, EngineDecodeBackend::Ort),
            WorkflowExecutionAdmission::CandidateTreeUnavailable { reason, .. }
                if reason.contains("tokens")
        ));
    }
}

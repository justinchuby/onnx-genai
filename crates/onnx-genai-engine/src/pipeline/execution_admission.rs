//! Canonical workflow execution-capability admission.

use onnx_genai_metadata::{
    CandidateTreeTopology, ComponentImplementation, DFlashStructure, InferenceMetadata,
    RuntimeInputRole, SemanticInputRole, SpeculativeAcceptedPath, SpeculativeContract,
    SpeculativeProposalExecution, StateFinalWriter, StatePortAccess, WorkflowSpec, WorkflowStep,
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
        let plan = Self {
            version: contract.version.clone(),
            proposer: contract.proposer.clone(),
            target: contract.target.clone(),
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
        };
        if let Some(reason) = candidate_tree_bypassed_step(
            &workflow.steps,
            "pipeline.workflow.steps",
            &plan.proposer,
            &plan.target,
        ) {
            return Err(reason);
        }
        if let Some(output) = workflow
            .outputs
            .keys()
            .find(|output| output.as_str() != onnx_genai_metadata::decoder_workflow::TOKENS_OUTPUT)
        {
            return Err(format!(
                "declared output '{output}' has no candidate-tree output-publication participant; \
                 the current driver stages only the declared token stream, so admitting it would \
                 skip its S4 family/site"
            ));
        }
        plan.validate_state_authority(workflow)?;
        let order = candidate_component_order(&workflow.steps, &plan.proposer, &plan.target);
        if order != [&plan.proposer, &plan.target] {
            return Err(format!(
                "candidate-tree flat execution requires proposer '{}' followed by target '{}', \
                 but authored invocation order is {order:?}; reorder the exact two invocations so \
                 specialized execution cannot change workflow ordering",
                plan.proposer, plan.target
            ));
        }
        Ok(plan)
    }

    fn validate_state_authority(&self, workflow: &WorkflowSpec) -> Result<(), String> {
        let state_plan = onnx_genai_metadata::resolve_state_plan(workflow);
        for (cell, _) in state_plan
            .session_cells()
            .filter(|(_, resolved)| resolved.transaction.required)
        {
            if !self.rollback_state.contains(cell) {
                return Err(format!(
                    "candidate-tree session state '{cell}' participates in the atomic turn but is \
                     absent from speculative.rollback_state"
                ));
            }
        }
        for cell in &self.rollback_state {
            if self
                .state_alias(workflow, &self.proposer, cell)
                .or_else(|| self.state_alias(workflow, &self.target, cell))
                .is_none()
            {
                return Err(format!(
                    "candidate-tree rollback participant '{cell}' has no proposer or target \
                     read-write state-service alias"
                ));
            }
            let resolved = state_plan.cell(cell).cloned().ok_or_else(|| {
                format!("candidate-tree rollback participant '{cell}' is unresolved")
            })?;
            let writer = resolved.final_writer.ok_or_else(|| {
                format!(
                    "candidate-tree rollback participant '{cell}' has no canonical final writer"
                )
            })?;
            let StateFinalWriter::Writer(writer) = writer else {
                return Err(format!(
                    "candidate-tree rollback participant '{cell}' uses session continuation; \
                     accepted-path state must be written by the declared proposer or target"
                ));
            };
            if writer.component.as_deref() != Some(self.proposer.as_str())
                && writer.component.as_deref() != Some(self.target.as_str())
            {
                return Err(format!(
                    "candidate-tree rollback participant '{cell}' final writer {:?} is outside \
                     the proposer/target transaction",
                    writer.component
                ));
            }
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
    if !workflow.effects.is_empty() {
        return Some(
            "effectful candidate-tree regions are unsupported until every external effect joins \
                     the accepted-path transaction"
                .to_string(),
        );
    }
    if !workflow
        .outputs
        .contains_key(onnx_genai_metadata::decoder_workflow::TOKENS_OUTPUT)
    {
        return Some(format!(
            "the workflow does not declare the '{}' output required for commit-only token \
                     publication",
            onnx_genai_metadata::decoder_workflow::TOKENS_OUTPUT
        ));
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
    for (component, ports) in [
        (&contract.proposer, &contract.port_bindings),
        (&contract.target, &contract.target_port_bindings),
    ] {
        let port = ports
            .get("context_tokens")
            .expect("checked candidate-tree context role above");
        let Some((inputs, _)) = component_invocation(&workflow.steps, component) else {
            continue;
        };
        let Some(binding) = inputs.get(port) else {
            return Some(format!(
                "candidate-tree context port '{component}::{port}' has no workflow binding"
            ));
        };
        if !matches!(
            workflow.inputs.get(binding).map(|input| &input.role),
            Some(SemanticInputRole::Runtime { role, .. })
                if *role == RuntimeInputRole::PromptTokens
        ) {
            return Some(format!(
                "candidate-tree context port '{component}::{port}' must bind the workflow's \
                         runtime prompt_tokens input, not '{binding}'"
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

/// Find authored steps the specialized tree drive would otherwise skip.
///
/// A sequence is only structural grouping, so recursively inspecting it admits
/// the same two-component template wherever it is grouped. A loop, branch,
/// unrelated invocation, or emit has ordering/effect/output semantics the
/// driver does not interpret; accepting one would make the metadata advisory.
fn candidate_tree_bypassed_step(
    steps: &[WorkflowStep],
    path: &str,
    proposer: &str,
    target: &str,
) -> Option<String> {
    for (index, step) in steps.iter().enumerate() {
        let step_path = format!("{path}[{index}]");
        match step {
            WorkflowStep::Invoke { component, .. }
                if component == proposer || component == target => {}
            WorkflowStep::Invoke { component, .. } => {
                return Some(format!(
                    "candidate-tree driver cannot execute unrelated component '{component}' at \
                     {step_path}; it would be skipped between proposer/target invocations"
                ));
            }
            WorkflowStep::Sequence { steps } => {
                if let Some(reason) = candidate_tree_bypassed_step(
                    steps,
                    &format!("{step_path}.steps"),
                    proposer,
                    target,
                ) {
                    return Some(reason);
                }
            }
            WorkflowStep::Loop { setup, .. } => {
                let detail = if setup.is_empty() {
                    "its condition/body ordering"
                } else {
                    "its non-empty setup, condition/body ordering"
                };
                return Some(format!(
                    "candidate-tree driver cannot faithfully execute loop at {step_path}: \
                     {detail} would be skipped; use a flat proposer/target invocation template \
                     until the generic workflow transaction hosts the candidate-tree seam"
                ));
            }
            WorkflowStep::Branch { .. } => {
                return Some(format!(
                    "candidate-tree driver cannot faithfully execute branch at {step_path}; its \
                     predicate/join would be skipped before accepted-prefix commit"
                ));
            }
            WorkflowStep::Emit { output, .. } => {
                return Some(format!(
                    "candidate-tree driver cannot publish emit at {step_path} into output \
                     '{output}'; its S4 family/site would be skipped"
                ));
            }
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

fn candidate_component_order<'a>(
    steps: &'a [WorkflowStep],
    proposer: &'a str,
    target: &'a str,
) -> Vec<&'a str> {
    let mut order = Vec::new();
    for step in steps {
        match step {
            WorkflowStep::Invoke { component, .. }
                if component == proposer || component == target =>
            {
                order.push(component.as_str());
            }
            WorkflowStep::Sequence { steps } => {
                order.extend(candidate_component_order(steps, proposer, target));
            }
            _ => {}
        }
    }
    order
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
